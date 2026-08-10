"""
StateMemory module.
Maintains a global JSON state file shared by orchestrator and agents.
"""

import json
import os
from typing import Any, Dict, List, Optional


class StateMemory:
    """State memory manager."""

    def __init__(self, memory_path: str = "./state_memory.json"):
        self.memory_path = memory_path
        self._state: Dict[str, Any] = {}

    def _default_state(
        self,
        personal_info: Dict,
        chief_complaint: str,
        present_illness: str,
        case_id: Optional[str],
    ) -> Dict[str, Any]:
        return {
            "case_id": case_id,
            "personal_info": personal_info,
            "chief_complaint": chief_complaint,
            "present_illness": present_illness,
            "dept_l1": None,
            "dept_l2": [],
            "past_history": None,
            "hypothesis_illness": [],
            "physical_exams": {},
            "auxiliary_exams": {},
            "imaging_results": {},
            "completed_physical_exams": [],
            "completed_auxiliary_exams": [],
            "completed_imaging_exams": [],
            "pending_auxiliary_exams": [],
            "predicted_physical_exams": [],
            "predicted_auxiliary_exams": [],
            "diagnosis_result": [],
            "treatment_plan": [],
            "task_status": {
                "task1_referral": "pending",
                "task2_doctor": "pending",
                "task3_imaging": "pending",
                "task4_diagnosis": "pending",
                "task5_treatment": "pending",
            },
            "recheck_total": 0,
            "max_recheck": 4,
            # MCTS fields
            "mcts_trajectory": [],
            "mcts_step": 0,
            "mcts_eta": 4,
            "workflow_action_history": [],
            "backtrack_events": [],
            "missing_evidence": [],
            "potential_missing_evidence": [],
            "experience_knowledge": [],
            "retrieved_guidelines": [],
            "retrieved_cdc_cases": [],
            "evidence_importance": {},
            "experience_memory_available": False,
            "top_hypothesis_confidence": 0.0,
            "mcts_snapshots": [],
            "active_snapshot_id": None,
            "terminated_by": None,
            "finalization_used": False,
        }

    def initialize(
        self,
        personal_info: Dict,
        chief_complaint: str,
        present_illness: str,
        case_id: Optional[str] = None,
    ):
        """Initialize working memory for a new case."""
        self._state = self._default_state(
            personal_info=personal_info,
            chief_complaint=chief_complaint,
            present_illness=present_illness,
            case_id=case_id,
        )
        self._save()
        return self._state

    def _save(self):
        """Persist state to JSON file."""
        with open(self.memory_path, "w", encoding="utf-8") as f:
            json.dump(self._state, f, ensure_ascii=False, indent=2)

    def _ensure_state_defaults(self):
        """Backfill missing keys for old state files."""
        defaults = self._default_state({}, "", "", None)
        changed = False
        for key, value in defaults.items():
            if key not in self._state:
                self._state[key] = value
                changed = True

        # Ensure task status keys exist
        task_defaults = defaults["task_status"]
        self._state.setdefault("task_status", {})
        for task, status in task_defaults.items():
            if task not in self._state["task_status"]:
                self._state["task_status"][task] = status
                changed = True

        if changed:
            self._save()

    def _load(self):
        """Load state from JSON file if it exists."""
        if os.path.exists(self.memory_path):
            with open(self.memory_path, "r", encoding="utf-8") as f:
                self._state = json.load(f)
            self._ensure_state_defaults()
        return self._state

    def get(self, key: str, default: Any = None) -> Any:
        """Get a state value by key."""
        self._load()
        return self._state.get(key, default)

    def set(self, key: str, value: Any):
        """Set a state value by key."""
        self._load()
        self._state[key] = value
        self._save()

    def update(self, updates: Dict[str, Any]):
        """Batch update state fields."""
        self._load()
        self._state.update(updates)
        self._save()

    def set_all(self, state: Dict[str, Any]):
        """Replace the full state object."""
        self._state = dict(state)
        self._ensure_state_defaults()
        self._save()

    def get_all(self) -> Dict:
        """Get full state."""
        self._load()
        return self._state.copy()

    # ==========================================
    # Task 1: Referral
    # ==========================================
    def set_department(self, dept_l1: str, dept_l2: List[str]):
        """Set department results and mark Task1 completed."""
        self._load()
        self._state["dept_l1"] = dept_l1
        self._state["dept_l2"] = dept_l2
        self._state["task_status"]["task1_referral"] = "completed"
        self._save()

    def update_department(self, dept_l1: str, dept_l2: List[str]):
        """Update department without changing Task1 status."""
        self._load()
        self._state["dept_l1"] = dept_l1
        self._state["dept_l2"] = dept_l2
        self._save()

    # ==========================================
    # Task 2: Doctor
    # ==========================================
    def set_past_history(self, past_history: str):
        """Set past history."""
        self._load()
        self._state["past_history"] = past_history
        self._save()

    def set_hypothesis_illness(self, hypothesis: List[Dict]):
        """Set differential hypotheses."""
        self._load()
        self._state["hypothesis_illness"] = hypothesis
        self._save()

    def add_physical_exam(self, exam_name: str, exam_result: Any):
        """Add a physical exam result."""
        self._load()
        self._state["physical_exams"][exam_name] = exam_result
        if exam_name not in self._state["completed_physical_exams"]:
            self._state["completed_physical_exams"].append(exam_name)
        self._save()

    def add_auxiliary_exam(self, exam_name: str, exam_result: Any):
        """Add an auxiliary exam result."""
        self._load()
        self._state["auxiliary_exams"][exam_name] = exam_result
        if exam_name not in self._state["completed_auxiliary_exams"]:
            self._state["completed_auxiliary_exams"].append(exam_name)
        self._save()

    def get_pending_physical_exams(self, required_exams: List[str]) -> List[str]:
        """Return required physical exams that are not completed yet."""
        self._load()
        completed = set(self._state.get("completed_physical_exams", []))
        return [e for e in required_exams if e not in completed]

    def get_pending_auxiliary_exams(self, required_exams: List[str]) -> List[str]:
        """Return required auxiliary exams that are not completed yet."""
        self._load()
        completed = set(self._state.get("completed_auxiliary_exams", []))
        return [e for e in required_exams if e not in completed]

    def mark_doctor_task_complete(self):
        """Mark Task2 as completed."""
        self._load()
        self._state["task_status"]["task2_doctor"] = "completed"
        self._save()

    # ==========================================
    # Task 3: Imaging
    # ==========================================
    def add_imaging_result(self, imaging_type: str, result: str):
        """Add an imaging result."""
        self._load()
        self._state["imaging_results"][imaging_type] = result
        if imaging_type not in self._state["completed_imaging_exams"]:
            self._state["completed_imaging_exams"].append(imaging_type)
        self._save()

    def get_pending_imaging_exams(self, required_exams: List[str]) -> List[str]:
        """Return required imaging exams that are not completed yet."""
        self._load()
        completed = set(self._state.get("completed_imaging_exams", []))
        return [e for e in required_exams if e not in completed]

    def mark_imaging_task_complete(self):
        """Mark Task3 as completed."""
        self._load()
        self._state["task_status"]["task3_imaging"] = "completed"
        self._save()

    # ==========================================
    # Task 4: Diagnosis
    # ==========================================
    def update_hypothesis_illness(self, hypothesis: List[Dict]):
        """Update differential hypotheses."""
        self._load()
        self._state["hypothesis_illness"] = hypothesis
        self._save()

    def set_diagnosis_result(self, diagnosis: List[str]):
        """Set diagnosis result and mark Task4 completed."""
        self._load()
        self._state["diagnosis_result"] = diagnosis
        self._state["task_status"]["task4_diagnosis"] = "completed"
        self._save()

    def increment_recheck(self) -> bool:
        """Increase recheck count if under limit. Return whether increment succeeded."""
        self._load()
        cur = int(self._state.get("recheck_total", 0) or 0)
        max_allowed = int(self._state.get("max_recheck", 0) or 0)
        if cur >= max_allowed:
            return False
        self._state["recheck_total"] = cur + 1
        self._save()
        return True

    def mark_diagnosis_need_recheck(self):
        """Mark diagnosis task as needing recheck."""
        self._load()
        self._state["task_status"]["task4_diagnosis"] = "need_recheck"
        self._save()

    # ==========================================
    # Task 5: Treatment
    # ==========================================
    def set_treatment_plan(self, treatment: List[str]):
        """Set treatment plan and mark Task5 completed."""
        self._load()
        self._state["treatment_plan"] = treatment
        self._state["task_status"]["task5_treatment"] = "completed"
        self._save()

    # ==========================================
    # Helpers
    # ==========================================
    def get_examination_summary(self) -> str:
        """Build a text summary of all collected exams."""
        self._load()
        physical = self._state.get("physical_exams", {})
        auxiliary = self._state.get("auxiliary_exams", {})
        imaging = self._state.get("imaging_results", {})

        result_parts = []

        if physical:
            parts = [f"{k}: {v}" for k, v in physical.items()]
            result_parts.append("[Physical Exam]\n" + "\n".join(parts))

        if auxiliary:
            parts = [f"{k}: {v}" for k, v in auxiliary.items()]
            result_parts.append("[Auxiliary Exam]\n" + "\n".join(parts))

        if imaging:
            parts = [f"{k}: {v}" for k, v in imaging.items()]
            result_parts.append("[Imaging Result]\n" + "\n".join(parts))

        return "\n\n".join(result_parts) if result_parts else "No exam results available"

    def get_hypothesis_summary(self) -> str:
        """Build a text summary of current hypotheses."""
        self._load()
        hypothesis = self._state.get("hypothesis_illness", [])
        if not hypothesis:
            return "No disease hypotheses available"

        parts = []
        for i, h in enumerate(hypothesis, 1):
            disease = h.get("disease") or h.get("name") or "Unknown"
            evidence = h.get("evidence", [])
            confidence = h.get("confidence", 0)
            evidence_text = ", ".join(evidence) if isinstance(evidence, list) else str(evidence)
            parts.append(
                f"{i}. {disease}\n"
                f"   Evidence: {evidence_text}\n"
                f"   Confidence: {confidence}"
            )

        return "\n".join(parts)

    def get_recheck_summary(self) -> Dict[str, Any]:
        """Return recheck stats."""
        self._load()
        return {"total": self._state.get("recheck_total", 0)}

    def is_task_completed(self, task_name: str) -> bool:
        """Check whether a task is completed."""
        self._load()
        return self._state.get("task_status", {}).get(task_name) == "completed"

    def reset(self):
        """Reset in-memory and on-disk state."""
        self._state = {}
        if os.path.exists(self.memory_path):
            os.remove(self.memory_path)
