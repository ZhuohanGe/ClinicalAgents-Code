import math
import random
from copy import deepcopy
from typing import Any, Callable, Dict, List, Optional, Tuple

from mcts_config import MCTSConfig
from mcts_prompts import (
    BACKTRACK_SYSTEM,
    BACKTRACK_USER,
    EXPANSION_SYSTEM,
    EXPANSION_USER,
    ROLLOUT_POLICY_SYSTEM,
    ROLLOUT_POLICY_USER,
    SIMULATION_SYSTEM,
    SIMULATION_USER,
    MISSING_EVIDENCE_SYSTEM,
    MISSING_EVIDENCE_USER,
)
from mcts_state import (
    append_backtrack_event,
    append_snapshot,
    choose_valid_backtrack_action,
    filter_valid_actions,
    record_action,
    recover_snapshot_state,
    resolve_backtrack_snapshot,
    snapshot_prompt_metadata,
)
from utils import parse_json_from_response


class MCTSNode:
    """Tree node = a snapshot of state dict."""

    def __init__(
        self,
        state_snapshot: Dict[str, Any],
        parent: Optional["MCTSNode"] = None,
        action: Optional[str] = None,
        prior: float = 0.0,
        immediate_reward: float = 0.0,
    ):
        self.state = state_snapshot
        self.parent = parent
        self.children: Dict[str, "MCTSNode"] = {}
        self.action = action
        self.visit_count = 0
        self.total_value = 0.0
        self.prior = prior
        self.immediate_reward = immediate_reward

    @property
    def q_value(self) -> float:
        return self.total_value / self.visit_count if self.visit_count > 0 else 0.0


class MCTSSearch:
    def __init__(
        self,
        planning_llm: Callable[[str, str], str],
        simulation_llm: Callable[[str, str], str],
        config: MCTSConfig,
        action_space: Dict[str, str],
    ):
        self.planning_llm = planning_llm
        self.simulation_llm = simulation_llm
        self.config = config
        self.action_space = action_space
        self.last_search_stats: Dict[str, Any] = {}
        self._remaining_rollout_calls: Optional[int] = None
        self._rollout_calls_used = 0

    def search(self, current_state: Dict[str, Any]) -> str:
        root = MCTSNode(deepcopy(current_state))
        self._rollout_calls_used = 0
        if int(getattr(self.config, "max_rollout_calls_per_search", 0) or 0) > 0:
            self._remaining_rollout_calls = int(self.config.max_rollout_calls_per_search)
        else:
            self._remaining_rollout_calls = None
        if self._is_terminal(root.state):
            self.last_search_stats = {
                "chosen_action": "a_term",
                "reason": "root_is_terminal",
                "children": [],
                "rollout_calls_used": 0,
                "transition_backend": "simulation_model",
            }
            return "a_term"

        root_actions = self._filter_actions(root.state)
        if len(root_actions) == 1:
            only_action = root_actions[0]
            self.last_search_stats = {
                "chosen_action": only_action,
                "reason": "single_valid_action",
                "children": [
                    {
                        "action": only_action,
                        "visit_count": 0,
                        "q_value": 0.0,
                        "prior": 1.0,
                        "total_value": 0.0,
                    }
                ],
                "rollout_calls_used": 0,
                "transition_backend": "simulation_model",
            }
            return only_action

        scored = self._score_actions(root.state, root_actions)
        if not scored:
            uniform = 1.0 / max(1, len(root_actions))
            scored = [(action, uniform) for action in root_actions]
        topk = sorted(scored, key=lambda item: item[1], reverse=True)[: max(1, self.config.K)]

        # Build every root candidate before any rollout so a budget can never
        # silently collapse Top-K evaluation into Top-1 evaluation.
        for action, prior in topk:
            root.children[action] = MCTSNode(
                state_snapshot=deepcopy(root.state),
                parent=root,
                action=action,
                prior=float(prior),
            )

        min_calls = self._minimum_rollout_call_budget(len(topk), root.state)
        configured_cap = int(getattr(self.config, "max_rollout_calls_per_search", 0) or 0)
        if configured_cap and configured_cap < min_calls:
            raise ValueError(
                "MCTS rollout call cap is too small for fair Top-K x N evaluation: "
                f"configured={configured_cap}, required_at_least={min_calls}. "
                "Set the cap to 0 (unlimited) or increase it."
            )

        actual_simulations = 0
        # Round-robin ordering gives every candidate the same number of
        # independent rollouts, as required by Eq. (6).
        for simulation_idx in range(self.config.N_sim):
            for action, prior in topk:
                aggregate_child = root.children[action]
                child_state, immediate_reward = self._transition(root.state, action)
                rollout_child = MCTSNode(
                    state_snapshot=child_state,
                    parent=aggregate_child,
                    action=action,
                    prior=float(prior),
                    immediate_reward=immediate_reward,
                )
                aggregate_child.children[f"rollout_{simulation_idx}"] = rollout_child
                leaf, value = self._rollout(rollout_child)
                previous_visits = aggregate_child.visit_count
                self._backpropagate(leaf, value)
                aggregate_child.immediate_reward = (
                    (
                        aggregate_child.immediate_reward
                        * previous_visits
                    )
                    + immediate_reward
                ) / max(1, previous_visits + 1)
                aggregate_child.state = child_state
                actual_simulations += 1

        best_action = self._select_root_action(root)
        if best_action is None:
            best_action = self._fallback_action(root.state, self._filter_actions(root.state))

        self.last_search_stats = {
            "chosen_action": best_action,
            "children": self._serialize_children(root),
            "simulations": actual_simulations,
            "expected_simulations": len(topk) * self.config.N_sim,
            "rollout_calls_used": self._rollout_calls_used,
            "minimum_fair_call_budget": min_calls,
            "transition_backend": "simulation_model",
        }
        return best_action

    def _minimum_rollout_call_budget(
        self,
        candidate_count: int,
        current_state: Optional[Dict[str, Any]] = None,
    ) -> int:
        """Return the minimum simulation-model calls for all promised rollouts."""
        # A normal transition uses state simulation + missing-evidence
        # verification. ``a_back`` additionally uses one backtrack-decision
        # call, so reserve the worst case to keep Top-K x N allocation fair.
        calls_per_transition = 3
        current_step = int((current_state or {}).get("mcts_step", 0) or 0)
        remaining_after_root = max(0, self.config.eta - (current_step + 1))
        rollout_depth = min(self.config.D, remaining_after_root)
        transitions_per_rollout = 1 + rollout_depth * max(1, self.config.K)
        return candidate_count * self.config.N_sim * calls_per_transition * transitions_per_rollout

    def _select(self, node: MCTSNode) -> MCTSNode:
        while node.children and not self._is_terminal(node.state):
            best_score, best_child = -float("inf"), None
            for child in node.children.values():
                score = self._paper_puct_score(child)
                if score > best_score:
                    best_score = score
                    best_child = child
            node = best_child if best_child is not None else node
            if best_child is None:
                break
        return node

    def _expand(self, node: MCTSNode) -> MCTSNode:
        if self._is_terminal(node.state):
            return node

        if node.children:
            return self._best_child_for_rollout(node)

        actions = self._filter_actions(node.state)
        scored = self._score_actions(node.state, actions)
        if not scored:
            uniform = 1.0 / max(1, len(actions))
            scored = [(a, uniform) for a in actions]

        topk = sorted(scored, key=lambda x: x[1], reverse=True)[: max(1, self.config.K)]
        for action, prior in topk:
            child_state, reward = self._transition(node.state, action)
            child = MCTSNode(
                state_snapshot=child_state,
                parent=node,
                action=action,
                prior=float(prior),
                immediate_reward=reward,
            )
            node.children[action] = child

        return self._best_child_for_rollout(node)

    def _rollout(self, node: MCTSNode) -> Tuple[MCTSNode, float]:
        current = node
        cumulative_reward = float(node.immediate_reward)
        discount = float(self.config.gamma_d)

        remaining_horizon = max(
            0,
            int(self.config.eta) - int(current.state.get("mcts_step", 0) or 0),
        )
        rollout_depth = min(self.config.D, remaining_horizon)
        for _ in range(rollout_depth):
            if self._is_terminal(current.state):
                break
            self._expand(current)
            selected = self._select(current)
            if selected is current:
                break
            cumulative_reward += discount * float(selected.immediate_reward)
            discount *= float(self.config.gamma_d)
            current = selected

        return current, cumulative_reward

    def _backpropagate(self, node: MCTSNode, value: float):
        cur = node
        while cur is not None:
            cur.visit_count += 1
            cur.total_value += value
            cur = cur.parent

    def _score_actions(self, state: Dict[str, Any], actions: List[str]) -> List[Tuple[str, float]]:
        if not actions:
            return []

        action_descriptions = "\n".join(
            [f"- {a}: {self.action_space.get(a, '')}" for a in actions]
        )
        user_prompt = EXPANSION_USER.format(
            personal_info=state.get("personal_info", {}),
            chief_complaint=state.get("chief_complaint", ""),
            present_illness=state.get("present_illness", ""),
            completed_physical_exams=state.get("completed_physical_exams", []),
            completed_auxiliary_exams=state.get("completed_auxiliary_exams", []),
            completed_imaging_exams=state.get("completed_imaging_exams", []),
            physical_exams=state.get("physical_exams", {}),
            auxiliary_exams=state.get("auxiliary_exams", {}),
            imaging_results=state.get("imaging_results", {}),
            hypothesis_illness=state.get("hypothesis_illness", []),
            dept_l1=state.get("dept_l1"),
            mcts_trajectory=state.get("mcts_trajectory", []),
            mcts_step=state.get("mcts_step", 0),
            experience_knowledge=state.get("experience_knowledge", []),
            potential_missing_evidence=state.get("potential_missing_evidence", []),
            action_descriptions=action_descriptions,
        )

        parsed = self._call_and_parse_json(self.planning_llm, EXPANSION_SYSTEM, user_prompt, call_kind="planning")
        if not isinstance(parsed, list):
            return []

        raw_map: Dict[str, float] = {}
        for item in parsed:
            if not isinstance(item, dict):
                continue
            action = str(item.get("action", "")).strip()
            if action not in actions:
                continue
            try:
                score = float(item.get("score", 0.0))
            except (TypeError, ValueError):
                score = 0.0
            raw_map[action] = max(0.0, score)

        raw_scores = []
        for action in actions:
            score = raw_map.get(action, 0.1)
            if state.get("dept_l1") is None and action == "task1_referral":
                score += 0.2
            raw_scores.append((action, score))

        return self._softmax(raw_scores)

    def _rollout_policy(self, state: Dict[str, Any]) -> str:
        actions = self._filter_actions(state)
        if not actions:
            return "a_term"

        user_prompt = ROLLOUT_POLICY_USER.format(
            chief_complaint=state.get("chief_complaint", ""),
            evidence_summary=self._build_evidence_summary(state),
            hypothesis_illness=state.get("hypothesis_illness", []),
            experience_knowledge=state.get("experience_knowledge", []),
            potential_missing_evidence=state.get("potential_missing_evidence", []),
            mcts_trajectory=state.get("mcts_trajectory", []),
            action_list=", ".join(actions),
        )
        resp = self._safe_call_model(
            self.simulation_llm,
            ROLLOUT_POLICY_SYSTEM,
            user_prompt,
            call_kind="rollout",
        )
        chosen = self._extract_action_from_text(resp, actions)
        return chosen if chosen else self._fallback_action(state, actions)

    def _simulate_action(self, state: Dict[str, Any], action: str) -> Dict[str, Any]:
        user_prompt = SIMULATION_USER.format(
            chief_complaint=state.get("chief_complaint", ""),
            present_illness=state.get("present_illness", ""),
            dept_l1=state.get("dept_l1", ""),
            dept_l2=state.get("dept_l2", []),
            evidence_summary=self._build_evidence_summary(state),
            hypothesis_illness=state.get("hypothesis_illness", []),
            pending_auxiliary_exams=state.get("pending_auxiliary_exams", []),
            task_status=state.get("task_status", {}),
            mcts_trajectory=state.get("mcts_trajectory", []),
            experience_knowledge=state.get("experience_knowledge", []),
            potential_missing_evidence=state.get("potential_missing_evidence", []),
            action=action,
            action_description=self.action_space.get(action, ""),
        )
        parsed = self._call_and_parse_json(
            self.simulation_llm,
            SIMULATION_SYSTEM,
            user_prompt,
            call_kind="rollout",
        )
        if not isinstance(parsed, dict):
            return {
                "new_evidence": {},
                "updated_hypotheses": [],
                "_simulation_valid": False,
            }
        parsed["_simulation_valid"] = True
        return parsed

    def _apply_simulation(
        self,
        state: Dict[str, Any],
        simulation: Dict[str, Any],
        action: str,
    ) -> Dict[str, Any]:
        sim_state = deepcopy(state)
        simulation_valid = simulation.get("_simulation_valid") is not False
        new_evidence = simulation.get("new_evidence", {}) or {}

        if isinstance(new_evidence, dict):
            evidence_to_completion = {
                "physical_exams": "completed_physical_exams",
                "auxiliary_exams": "completed_auxiliary_exams",
                "imaging_results": "completed_imaging_exams",
            }
            for evidence_field, completed_field in evidence_to_completion.items():
                predicted = new_evidence.get(evidence_field, {}) or {}
                if not isinstance(predicted, dict):
                    continue
                current = dict(sim_state.get(evidence_field, {}) or {})
                current.update(predicted)
                sim_state[evidence_field] = current
                completed = list(sim_state.get(completed_field, []) or [])
                for exam_name in predicted:
                    if exam_name not in completed:
                        completed.append(exam_name)
                sim_state[completed_field] = completed

        hyps = self._normalize_hypotheses(simulation.get("updated_hypotheses", []))
        if hyps:
            sim_state["hypothesis_illness"] = hyps

        state_updates = simulation.get("state_updates", {}) or {}
        if isinstance(state_updates, dict):
            scalar_fields = {"dept_l1"}
            list_fields = {
                "dept_l2",
                "predicted_physical_exams",
                "predicted_auxiliary_exams",
                "pending_auxiliary_exams",
                "diagnosis_result",
                "treatment_plan",
                "experience_knowledge",
                "potential_missing_evidence",
            }
            for field in scalar_fields:
                value = state_updates.get(field)
                if isinstance(value, str) and value.strip():
                    sim_state[field] = value.strip()
            for field in list_fields:
                value = state_updates.get(field)
                if isinstance(value, list):
                    sim_state[field] = deepcopy(value)

        if simulation_valid and action.startswith("task"):
            status = dict(sim_state.get("task_status", {}) or {})
            status[action] = "completed"
            sim_state["task_status"] = status
        if not simulation_valid:
            sim_state["transition_error"] = action
        sim_state["rollout_transition_source"] = "simulation_model"

        return sim_state

    def _transition(self, state: Dict[str, Any], action: str) -> Tuple[Dict[str, Any], float]:
        """Sample M_work^{t+1} for an action and compute Eq. dense reward."""
        old_missing = list(state.get("missing_evidence", []) or [])
        old_top_conf = self._top_confidence(state)

        sim_state = append_snapshot(deepcopy(state), reason="rollout_before_%s" % action)

        if action == "a_term":
            sim_state = record_action(sim_state, action, self.action_space, source="rollout")
            return sim_state, 0.0
        if self._remaining_rollout_calls is not None and self._remaining_rollout_calls <= 0:
            sim_state["missing_evidence"] = old_missing
            sim_state["top_hypothesis_confidence"] = old_top_conf
            return sim_state, 0.0

        if action == "a_back":
            sim_state = self._simulate_backtrack_transition(sim_state)
        else:
            simulation = self._simulate_action(sim_state, action)
            sim_state = self._apply_simulation(sim_state, simulation, action)
            sim_state = record_action(sim_state, action, self.action_space, source="rollout")

        verification_failed = False
        if self._remaining_rollout_calls is not None and self._remaining_rollout_calls <= 0:
            new_missing = old_missing
        else:
            detected_missing = self._detect_missing_evidence(sim_state)
            verification_failed = detected_missing is None
            new_missing = old_missing if verification_failed else detected_missing
        sim_state["missing_evidence"] = new_missing
        sim_state["missing_evidence_verification_failed"] = verification_failed
        new_top_conf = self._top_confidence(sim_state)
        sim_state["top_hypothesis_confidence"] = new_top_conf

        delta_e = len(old_missing) - len(new_missing)
        delta_c = new_top_conf - old_top_conf
        reward = self._dense_reward(delta_e, delta_c)
        if sim_state.get("missing_evidence_verification_failed"):
            reward -= float(self.config.gamma)
        if sim_state.get("transition_error"):
            reward -= float(self.config.gamma)
        return sim_state, reward

    def _simulate_backtrack_transition(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Simulate the same restore-then-act semantics used by real a_back."""
        workflow_history = list(state.get("workflow_action_history", []) or [])
        before_index = len(workflow_history) - 1 if workflow_history else None
        metadata = snapshot_prompt_metadata(state, before_workflow_action_index=before_index)
        prompt = BACKTRACK_USER.format(
            hypothesis_illness=state.get("hypothesis_illness", []),
            missing_evidence=state.get("missing_evidence", []),
            mcts_trajectory=state.get("mcts_trajectory", []),
            mcts_step=state.get("mcts_step", 0),
            snapshot_metadata=metadata,
        )
        decision = self._call_and_parse_json(
            self.simulation_llm, BACKTRACK_SYSTEM, prompt, call_kind="rollout"
        )
        decision_failed = not isinstance(decision, dict)
        if decision_failed:
            decision = {}
        target_snapshot = resolve_backtrack_snapshot(
            state, decision, before_workflow_action_index=before_index
        )
        back_state = record_action(
            state, "a_back", self.action_space, source="rollout_backtrack"
        )
        if target_snapshot is None:
            back_state["transition_error"] = "a_back:no_eligible_snapshot"
            return back_state
        recovered = recover_snapshot_state(back_state, target_snapshot)
        requested_action = decision.get("target_action")
        target_action = choose_valid_backtrack_action(
            recovered, requested_action, self.action_space, eta=self.config.eta
        )
        recovered = append_backtrack_event(
            recovered,
            target_snapshot,
            requested_action=requested_action,
            executed_action=target_action,
            source="rollout",
        )
        if target_action is None:
            recovered["transition_error"] = "a_back:no_valid_target_action"
            return recovered
        recovered = append_snapshot(
            recovered, reason="rollout_backtrack_restored",
            workflow_stage=target_snapshot["workflow_stage"]
        )
        simulation = self._simulate_action(recovered, target_action)
        recovered = self._apply_simulation(recovered, simulation, target_action)
        recovered = record_action(
            recovered, target_action, self.action_space,
            source="rollout_backtrack_target"
        )
        if decision_failed:
            recovered["transition_error"] = "a_back:invalid_decision"
        return recovered

    def _dense_reward(self, delta_e: float, delta_c: float) -> float:
        evidence_gain = max(0.0, float(delta_e))
        confidence_gain = max(0.0, float(delta_c))
        reward = (
            float(self.config.alpha) * evidence_gain
            + float(self.config.beta) * confidence_gain
        )
        if delta_e <= 0 and delta_c <= 0:
            reward -= float(self.config.gamma)
        return reward

    def _detect_missing_evidence(
        self,
        state: Dict[str, Any],
    ) -> Optional[List[str]]:
        user_prompt = MISSING_EVIDENCE_USER.format(
            chief_complaint=state.get("chief_complaint", ""),
            physical_exams=state.get("physical_exams", {}),
            auxiliary_exams=state.get("auxiliary_exams", {}),
            imaging_results=state.get("imaging_results", {}),
            past_history=state.get("past_history", ""),
            hypothesis_illness=state.get("hypothesis_illness", []),
            experience_knowledge=state.get("experience_knowledge", []),
            potential_missing_evidence=state.get("potential_missing_evidence", []),
        )
        parsed = self._call_and_parse_json(
            self.simulation_llm,
            MISSING_EVIDENCE_SYSTEM,
            user_prompt,
            call_kind="rollout",
        )
        if isinstance(parsed, list):
            detected = [str(x).strip() for x in parsed if str(x).strip()]
        else:
            return None
        return self._merge_text_items(detected)

    def _filter_actions(self, state: Dict[str, Any]) -> List[str]:
        return filter_valid_actions(state, self.action_space, eta=self.config.eta)

    def _is_terminal(self, state: Dict[str, Any]) -> bool:
        trajectory = state.get("mcts_trajectory", []) or []
        if trajectory and trajectory[-1] == "a_term":
            return True
        if int(state.get("mcts_step", 0) or 0) >= int(
            state.get("mcts_eta", self.config.eta) or self.config.eta
        ):
            return True
        return bool(state.get("diagnosis_result")) and bool(state.get("treatment_plan"))

    def _select_root_action(self, root: MCTSNode) -> Optional[str]:
        if not root.children:
            return None
        ranked = sorted(
            root.children.values(),
            key=lambda c: (self._paper_puct_score(c), c.q_value, c.prior, -c.visit_count),
            reverse=True,
        )
        return ranked[0].action if ranked else None

    def _best_child_for_rollout(self, node: MCTSNode) -> MCTSNode:
        if not node.children:
            return node
        ranked = sorted(
            node.children.values(),
            key=lambda c: (self._paper_puct_score(c), c.prior, -c.visit_count),
            reverse=True,
        )
        return ranked[0]

    def _paper_puct_score(self, node: MCTSNode) -> float:
        # During expansion every child already has one sampled immediate
        # transition, but it has not yet received a rollout/backpropagation.
        # Use that observation as its temporary Q without inventing a visit;
        # otherwise the first real simulation would be double-counted.
        q_estimate = node.q_value if node.visit_count > 0 else node.immediate_reward
        return float(q_estimate) + float(self.config.lambda_puct) * float(node.prior)

    def _fallback_action(self, state: Dict[str, Any], actions: List[str]) -> str:
        if not actions:
            return "a_term"
        if state.get("dept_l1") is None and "task1_referral" in actions:
            return "task1_referral"
        return random.choice(actions)

    def _serialize_children(self, node: MCTSNode) -> List[Dict[str, Any]]:
        items = []
        for action, child in node.children.items():
            items.append(
                {
                    "action": action,
                    "visit_count": child.visit_count,
                    "q_value": round(child.q_value, 4),
                    "prior": round(child.prior, 4),
                    "total_value": round(child.total_value, 4),
                    "puct_score": round(self._paper_puct_score(child), 4),
                    "immediate_reward": round(child.immediate_reward, 4),
                }
            )
        return sorted(items, key=lambda x: x["visit_count"], reverse=True)

    def _normalize_hypotheses(self, hypotheses: Any) -> List[Dict[str, Any]]:
        if not isinstance(hypotheses, list):
            return []

        normalized: List[Dict[str, Any]] = []
        for h in hypotheses:
            if isinstance(h, str):
                name = h.strip()
                if not name:
                    continue
                normalized.append({"name": name, "disease": name, "evidence": [], "confidence": 0.5})
                continue
            if not isinstance(h, dict):
                continue

            name = str(h.get("name") or h.get("disease") or "").strip()
            if not name:
                continue
            evidence = h.get("evidence", [])
            if isinstance(evidence, str):
                evidence = [evidence]
            if not isinstance(evidence, list):
                evidence = []
            try:
                confidence = float(h.get("confidence", 0.0))
            except (TypeError, ValueError):
                confidence = 0.0
            confidence = max(0.0, min(1.0, confidence))
            normalized.append(
                {
                    "name": name,
                    "disease": name,
                    "evidence": [str(x) for x in evidence if str(x).strip()],
                    "confidence": confidence,
                }
            )
        return normalized

    def _top_confidence(self, state: Dict[str, Any]) -> float:
        hyps = self._normalize_hypotheses(state.get("hypothesis_illness", []))
        if not hyps:
            return 0.0
        return max(float(h.get("confidence", 0.0)) for h in hyps)

    def _softmax(self, action_scores: List[Tuple[str, float]]) -> List[Tuple[str, float]]:
        if not action_scores:
            return []
        values = [score for _, score in action_scores]
        max_v = max(values)
        exps = [math.exp(v - max_v) for v in values]
        z = sum(exps) if exps else 1.0
        return [(action_scores[i][0], exps[i] / z) for i in range(len(action_scores))]

    def _extract_action_from_text(self, text: str, actions: List[str]) -> Optional[str]:
        if not text:
            return None
        clean = str(text).strip()
        if clean in actions:
            return clean
        for action in actions:
            if action in clean:
                return action
        return None

    def _build_evidence_summary(self, state: Dict[str, Any]) -> str:
        parts = []
        for label, field in (
            ("Physical", "physical_exams"),
            ("Auxiliary", "auxiliary_exams"),
            ("Imaging", "imaging_results"),
        ):
            value = state.get(field, {})
            if value:
                parts.append(f"{label}: {value}")
        knowledge = state.get("experience_knowledge", [])
        potential = state.get("potential_missing_evidence", [])
        if knowledge:
            parts.append(f"Experience memory: {knowledge}")
        if potential:
            parts.append(f"Potential missing evidence: {potential}")
        return " | ".join(parts) if parts else "No collected evidence"

    def _merge_text_items(self, *groups: Any) -> List[str]:
        merged: List[str] = []
        seen = set()
        for group in groups:
            if not group:
                continue
            items = group if isinstance(group, list) else [group]
            for item in items:
                text = str(item).strip()
                if not text or text in seen:
                    continue
                seen.add(text)
                merged.append(text)
        return merged

    def _safe_call_model(
        self,
        model: Callable[[str, str], str],
        system: str,
        user: str,
        call_kind: str = "generic",
    ) -> str:
        if call_kind == "rollout":
            if self._remaining_rollout_calls is not None:
                if self._remaining_rollout_calls <= 0:
                    return ""
                self._remaining_rollout_calls -= 1
            self._rollout_calls_used += 1
        try:
            return model(system, user)
        except Exception:
            return ""

    def _call_and_parse_json(
        self,
        model: Callable[[str, str], str],
        system: str,
        user: str,
        call_kind: str = "generic",
    ) -> Any:
        resp_1 = self._safe_call_model(model, system, user, call_kind=call_kind)
        parsed_1 = parse_json_from_response(resp_1)
        if parsed_1 is not None:
            return parsed_1

        # A retry would make the number of calls per rollout data-dependent and
        # can violate equal N-rollout allocation. Rollout parse failures instead
        # produce an empty transition and receive the uninformative penalty.
        if call_kind == "rollout":
            return None

        resp_2 = self._safe_call_model(model, system, user, call_kind=call_kind)
        parsed_2 = parse_json_from_response(resp_2)
        return parsed_2
