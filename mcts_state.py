"""Pure helpers for MCTS action counting, snapshots, and backtracking."""
from copy import deepcopy
from typing import Any, Dict, List, Optional, Sequence


TASK_ACTIONS = (
    "task1_referral",
    "task2_doctor",
    "task3_imaging",
    "task4_diagnosis",
    "task5_treatment",
)
CONTROL_ACTIONS = ("a_rag", "a_back", "a_term")


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def infer_workflow_stage(state: Dict[str, Any]) -> str:
    """Infer the latest completed clinical stage from a state snapshot."""
    if state.get("treatment_plan"):
        return "task5_treatment"
    if state.get("diagnosis_result"):
        return "task4_diagnosis"
    if state.get("completed_imaging_exams") or state.get("imaging_results"):
        return "task3_imaging"
    if state.get("hypothesis_illness"):
        return "task2_doctor"
    if state.get("dept_l1"):
        return "task1_referral"
    return "initial"


def record_action(
    state: Dict[str, Any],
    action: str,
    action_space: Sequence[str],
    source: str,
    count_mcts: bool = True,
) -> Dict[str, Any]:
    """Record one real or simulated action without mixing in marker strings."""
    if action not in action_space:
        raise ValueError("Unknown orchestration action: %s" % action)

    next_state = deepcopy(state)
    trajectory = [
        item
        for item in list(next_state.get("mcts_trajectory", []) or [])
        if item in action_space
    ]
    if count_mcts:
        trajectory.append(action)
    next_state["mcts_trajectory"] = trajectory
    next_state["mcts_step"] = len(trajectory)

    history = list(next_state.get("workflow_action_history", []) or [])
    history.append(
        {
            "sequence_index": len(history),
            "action": action,
            "source": source,
            "mcts_step": len(trajectory),
            "workflow_stage": infer_workflow_stage(next_state),
        }
    )
    next_state["workflow_action_history"] = history
    return next_state


def _normalize_snapshot_entry(entry: Any, index: int) -> Dict[str, Any]:
    if isinstance(entry, dict) and isinstance(entry.get("state"), dict):
        snapshot_state = deepcopy(entry["state"])
        snapshot_id = _safe_int(entry.get("snapshot_id"), index)
        trajectory_index = _safe_int(
            entry.get("trajectory_index"),
            len(snapshot_state.get("mcts_trajectory", []) or []),
        )
        workflow_action_index = _safe_int(
            entry.get("workflow_action_index"),
            len(snapshot_state.get("workflow_action_history", []) or []),
        )
        mcts_step = _safe_int(entry.get("mcts_step"), trajectory_index)
        workflow_stage = str(
            entry.get("workflow_stage") or infer_workflow_stage(snapshot_state)
        )
        reason = str(entry.get("reason") or "legacy_snapshot")
    else:
        snapshot_state = deepcopy(entry) if isinstance(entry, dict) else {}
        snapshot_id = index
        trajectory_index = len(snapshot_state.get("mcts_trajectory", []) or [])
        workflow_action_index = len(
            snapshot_state.get("workflow_action_history", []) or []
        )
        mcts_step = _safe_int(snapshot_state.get("mcts_step"), trajectory_index)
        workflow_stage = infer_workflow_stage(snapshot_state)
        reason = "legacy_snapshot"

    snapshot_state["mcts_snapshots"] = []
    return {
        "snapshot_id": snapshot_id,
        "trajectory_index": trajectory_index,
        "workflow_action_index": workflow_action_index,
        "mcts_step": mcts_step,
        "workflow_stage": workflow_stage,
        "reason": reason,
        "state": snapshot_state,
    }


def normalize_snapshots(raw_snapshots: Any) -> List[Dict[str, Any]]:
    """Convert old raw-state snapshots into metadata-wrapped snapshots."""
    items = raw_snapshots if isinstance(raw_snapshots, list) else []
    normalized: List[Dict[str, Any]] = []
    used_ids = set()
    next_id = 0
    for index, item in enumerate(items):
        snapshot = _normalize_snapshot_entry(item, index)
        snapshot_id = snapshot["snapshot_id"]
        if snapshot_id in used_ids:
            while next_id in used_ids:
                next_id += 1
            snapshot_id = next_id
            snapshot["snapshot_id"] = snapshot_id
        used_ids.add(snapshot_id)
        next_id = max(next_id, snapshot_id + 1)
        normalized.append(snapshot)
    return normalized


def append_snapshot(
    state: Dict[str, Any],
    reason: str,
    workflow_stage: Optional[str] = None,
) -> Dict[str, Any]:
    """Append an immutable snapshot with a stable ID and explicit coordinates."""
    next_state = deepcopy(state)
    snapshots = normalize_snapshots(next_state.get("mcts_snapshots", []))
    snapshot_id = max(
        [item["snapshot_id"] for item in snapshots] + [-1]
    ) + 1
    snapshot_state = deepcopy(next_state)
    snapshot_state["mcts_snapshots"] = []
    trajectory = list(next_state.get("mcts_trajectory", []) or [])
    workflow_history = list(next_state.get("workflow_action_history", []) or [])
    snapshots.append(
        {
            "snapshot_id": snapshot_id,
            "trajectory_index": len(trajectory),
            "workflow_action_index": len(workflow_history),
            "mcts_step": len(trajectory),
            "workflow_stage": workflow_stage or infer_workflow_stage(next_state),
            "reason": reason,
            "state": snapshot_state,
        }
    )
    next_state["mcts_snapshots"] = snapshots
    return next_state


def snapshot_prompt_metadata(
    state: Dict[str, Any],
    before_workflow_action_index: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Return compact, stable snapshot coordinates for the backtrack prompt."""
    snapshots = eligible_backtrack_snapshots(
        state,
        before_workflow_action_index=before_workflow_action_index,
    )
    metadata: List[Dict[str, Any]] = []
    for snapshot in snapshots:
        snap_state = snapshot["state"]
        hypotheses = snap_state.get("hypothesis_illness", []) or []
        hypothesis_names = []
        for item in hypotheses[:5]:
            if isinstance(item, dict):
                name = item.get("disease") or item.get("name")
            else:
                name = item
            if name:
                hypothesis_names.append(str(name))
        metadata.append(
            {
                "snapshot_id": snapshot["snapshot_id"],
                "workflow_stage": snapshot["workflow_stage"],
                "trajectory_index": snapshot["trajectory_index"],
                "workflow_action_index": snapshot["workflow_action_index"],
                "mcts_step": snapshot["mcts_step"],
                "reason": snapshot["reason"],
                "department": snap_state.get("dept_l1"),
                "hypotheses": hypothesis_names,
                "pending_auxiliary_exams": snap_state.get(
                    "pending_auxiliary_exams", []
                ),
                "has_diagnosis": bool(snap_state.get("diagnosis_result")),
            }
        )
    return metadata


def eligible_backtrack_snapshots(
    state: Dict[str, Any],
    before_workflow_action_index: Optional[int] = None,
) -> List[Dict[str, Any]]:
    snapshots = normalize_snapshots(state.get("mcts_snapshots", []))
    if before_workflow_action_index is None:
        history = list(state.get("workflow_action_history", []) or [])
        if not history:
            # Legacy states had no workflow history. Their last snapshot was
            # the current state, so only earlier list entries are eligible.
            return snapshots[:-1] if len(snapshots) > 1 else []
        before_workflow_action_index = len(history) - 1
    return [
        snapshot
        for snapshot in snapshots
        if snapshot["workflow_action_index"] <= before_workflow_action_index
    ]


def resolve_backtrack_snapshot(
    state: Dict[str, Any],
    decision: Dict[str, Any],
    before_workflow_action_index: Optional[int] = None,
) -> Optional[Dict[str, Any]]:
    """Resolve a decision by stable metadata, never by raw list position."""
    eligible = eligible_backtrack_snapshots(
        state,
        before_workflow_action_index=before_workflow_action_index,
    )
    if not eligible:
        return None

    requested_id = decision.get("target_snapshot_id")
    if requested_id is not None:
        requested_id = _safe_int(requested_id, -1)
        for snapshot in eligible:
            if snapshot["snapshot_id"] == requested_id:
                return snapshot
        # An explicit but unknown stable ID must never degrade into a raw
        # position/stage fallback, which could restore an unrelated state.
        return None

    requested_stage = str(
        decision.get("target_workflow_stage")
        or decision.get("workflow_stage")
        or ""
    ).strip()
    if requested_stage:
        stage_matches = [
            snapshot
            for snapshot in eligible
            if snapshot["workflow_stage"] == requested_stage
        ]
        if stage_matches:
            return stage_matches[-1]

    if decision.get("target_trajectory_index") is not None:
        target_index = _safe_int(decision.get("target_trajectory_index"), -1)
        index_matches = [
            snapshot
            for snapshot in eligible
            if snapshot["trajectory_index"] <= target_index
        ]
        if index_matches:
            return index_matches[-1]

    # Backward-compatible target_step is interpreted as an MCTS step value,
    # not as a Python list index.
    if decision.get("target_step") is not None:
        target_step = _safe_int(decision.get("target_step"), -1)
        step_matches = [
            snapshot
            for snapshot in eligible
            if snapshot["mcts_step"] <= target_step
        ]
        if step_matches:
            return step_matches[-1]

    return eligible[-1]


def recover_snapshot_state(
    current_state: Dict[str, Any],
    snapshot: Dict[str, Any],
) -> Dict[str, Any]:
    """Restore clinical state while preserving monotonic orchestration history."""
    recovered = deepcopy(snapshot["state"])
    preserved_fields = (
        "mcts_snapshots",
        "mcts_trajectory",
        "mcts_step",
        "mcts_eta",
        "workflow_action_history",
        "backtrack_events",
        "recheck_total",
        "max_recheck",
        "experience_memory_available",
        "terminated_by",
        "finalization_used",
    )
    for field in preserved_fields:
        if field in current_state:
            recovered[field] = deepcopy(current_state[field])
    recovered["active_snapshot_id"] = snapshot["snapshot_id"]
    return recovered


def append_backtrack_event(
    state: Dict[str, Any],
    snapshot: Dict[str, Any],
    requested_action: Optional[str],
    executed_action: Optional[str],
    source: str,
) -> Dict[str, Any]:
    next_state = deepcopy(state)
    events = list(next_state.get("backtrack_events", []) or [])
    events.append(
        {
            "event_id": len(events),
            "source": source,
            "target_snapshot_id": snapshot["snapshot_id"],
            "target_workflow_stage": snapshot["workflow_stage"],
            "requested_action": requested_action,
            "executed_action": executed_action,
            "mcts_step": _safe_int(next_state.get("mcts_step"), 0),
        }
    )
    next_state["backtrack_events"] = events
    return next_state


def filter_valid_actions(
    state: Dict[str, Any],
    action_space: Dict[str, str],
    eta: Optional[int] = None,
    for_backtrack_target: bool = False,
) -> List[str]:
    """Return state-valid actions for both search and committed backtracking."""
    actions = list(action_space.keys())
    diagnosis = state.get("diagnosis_result", []) or []
    treatment = state.get("treatment_plan", []) or []
    if diagnosis and treatment:
        return [] if for_backtrack_target else ["a_term"]

    max_steps = _safe_int(
        eta if eta is not None else state.get("mcts_eta"),
        0,
    )
    current_step = _safe_int(state.get("mcts_step"), 0)
    if max_steps > 0 and current_step >= max_steps:
        return []

    dept = state.get("dept_l1")
    hypotheses = state.get("hypothesis_illness", []) or []
    pending_exams = state.get("pending_auxiliary_exams", []) or []
    task_status = state.get("task_status", {}) or {}

    if not dept:
        actions = [
            action
            for action in actions
            if action not in (
                "task2_doctor",
                "task3_imaging",
                "task4_diagnosis",
                "task5_treatment",
            )
        ]
    if not hypotheses:
        actions = [
            action
            for action in actions
            if action not in (
                "task3_imaging",
                "task4_diagnosis",
                "task5_treatment",
            )
        ]
    if not pending_exams:
        actions = [action for action in actions if action != "task3_imaging"]
    if not diagnosis:
        actions = [action for action in actions if action != "task5_treatment"]

    for task_name in TASK_ACTIONS:
        if task_status.get(task_name) == "completed" and task_name in actions:
            actions.remove(task_name)

    actions = [action for action in actions if action != "a_term"]
    if not state.get("experience_memory_available", False):
        actions = [action for action in actions if action != "a_rag"]

    recheck_exhausted = _safe_int(state.get("recheck_total"), 0) >= _safe_int(
        state.get("max_recheck"), 0
    )
    remaining_steps = max_steps - current_step if max_steps > 0 else None
    if (
        not state.get("missing_evidence")
        or not eligible_backtrack_snapshots(state)
        or recheck_exhausted
        or (remaining_steps is not None and remaining_steps < 2)
    ):
        actions = [action for action in actions if action != "a_back"]

    if for_backtrack_target:
        return [action for action in actions if action not in ("a_back", "a_term")]

    if not actions:
        if diagnosis and "task5_treatment" in action_space:
            return ["task5_treatment"]
        if hypotheses and "task4_diagnosis" in action_space:
            return ["task4_diagnosis"]
        if dept and "task2_doctor" in action_space:
            return ["task2_doctor"]
        return ["task1_referral"] if "task1_referral" in action_space else []
    return actions


def choose_valid_backtrack_action(
    state: Dict[str, Any],
    requested_action: Optional[str],
    action_space: Dict[str, str],
    eta: Optional[int] = None,
) -> Optional[str]:
    valid_actions = filter_valid_actions(
        state,
        action_space,
        eta=eta,
        for_backtrack_target=True,
    )
    if requested_action in valid_actions:
        return requested_action
    return valid_actions[0] if valid_actions else None
