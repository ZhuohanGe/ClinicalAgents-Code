"""Medical orchestration controller for multi-agent diagnosis."""
from copy import deepcopy
from typing import Any, Callable, Dict, List, Optional, Tuple
from app_config import (
    EXPERIENCE_CDC_PATH,
    EXPERIENCE_GUIDE_PATH,
    FORCE_FINALIZATION,
    LLM_CONFIG,
    MAX_RECHECK_PER_CASE,
    MCTS_PLANNING_CONFIG,
    MCTS_ROLLOUT_CALL_CAP_PER_SEARCH,
    MCTS_ROLLOUT_CALL_CAP_TOTAL,
    MCTS_ROLLOUT_CONFIG,
    MCTS_WARM_START,
)
from state_memory import StateMemory
from experience_memory import ExperienceMemory
from agents import (
    ReferralAgent, ReferralVerifier,
    DoctorAgent, ImagingAgent,
    DiagnosisAgent, TreatmentAgent
)
from agents.doctor_agent import VALID_PHYSICAL_EXAMS, VALID_AUXILIARY_EXAMS
from utils import (
    load_prompt,
    call_llm_api,
    call_llm_api_with_config,
    parse_json_from_response,
    img_api,
    llm_phase,
    set_case_llm_phase,
)
from mcts_config import MCTSConfig
from mcts_search import MCTSSearch
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
from mcts_prompts import (
    BACKTRACK_SYSTEM,
    BACKTRACK_USER,
    MISSING_EVIDENCE_SYSTEM,
    MISSING_EVIDENCE_USER,
    HYPOTHESIS_CONFIDENCE_SYSTEM,
    HYPOTHESIS_CONFIDENCE_USER,
)


ACTION_SPACE = {
    "task1_referral": "Triage/referral agent",
    "task2_doctor": "Doctor agent",
    "task3_imaging": "Imaging and auxiliary exam agent",
    "task4_diagnosis": "Diagnosis synthesis agent",
    "task5_treatment": "Treatment planning agent",
    "a_rag": "Retrieve clinical memory/guidelines",
    "a_back": "Backtrack to prior step",
    "a_term": "Terminate orchestration",
}


class MedOrchestrator:
    """Medical task orchestrator."""
    def __init__(
        self,
        memory_path: str = "./state_memory.json",
        img_base_dir: str = './datasets/MedImg/',
        mcts_planning_config: Optional[Dict[str, str]] = None,
        mcts_rollout_config: Optional[Dict[str, str]] = None,
        mcts_rollout_call_cap_per_search: int = MCTS_ROLLOUT_CALL_CAP_PER_SEARCH,
        mcts_rollout_call_cap_total: int = MCTS_ROLLOUT_CALL_CAP_TOTAL,
        max_recheck: int = MAX_RECHECK_PER_CASE,
        experience_guide_path: str = EXPERIENCE_GUIDE_PATH,
        experience_cdc_path: str = EXPERIENCE_CDC_PATH,
        held_out_case_ids: Optional[List[str]] = None,
        guide_retriever: Optional[
            Callable[[str, List[Dict[str, Any]], int], List[Dict[str, Any]]]
        ] = None,
        force_finalization: bool = FORCE_FINALIZATION,
        warm_start: bool = MCTS_WARM_START,
    ):
        """Initialize agents, memory, and orchestration settings."""
        self.mcts_planning_config = dict(mcts_planning_config or MCTS_PLANNING_CONFIG)
        self.mcts_rollout_config = dict(mcts_rollout_config or MCTS_ROLLOUT_CONFIG)
        self._assert_distinct_rollout_model(LLM_CONFIG, self.mcts_rollout_config)
        self.memory = StateMemory(memory_path)
        self.img_base_dir = img_base_dir
        # Zero is the documented paper-mode default: no artificial rollout cap.
        # A positive cap is validated by MCTSSearch before any candidate runs.
        self.mcts_rollout_call_cap_per_search = max(
            0, int(mcts_rollout_call_cap_per_search or 0)
        )
        self.mcts_rollout_call_cap_total = max(0, int(mcts_rollout_call_cap_total or 0))
        self.max_recheck = max(0, int(max_recheck or 0))
        self.force_finalization = bool(force_finalization)
        self.warm_start = bool(warm_start)
        self.experience_memory = ExperienceMemory(
            guide_path=experience_guide_path,
            cdc_path=experience_cdc_path,
            held_out_case_ids=held_out_case_ids,
            allow_dataset_fallback=False,
            guide_retriever=guide_retriever,
        ).freeze()

        # Orchestration processing step.
        self.referral_agent = ReferralAgent()
        self.doctor_agent = DoctorAgent()
        self.imaging_agent = ImagingAgent(img_base_dir)
        self.diagnosis_agent = DiagnosisAgent()
        self.treatment_agent = TreatmentAgent()

        # Orchestration processing step.
        self.decision_prompt = load_prompt("orchestrator_decision")
        self.dept_verify_prompt = load_prompt("dept_verify")
        self.dept_final_prompt = load_prompt("dept_final")
        self.doctor_recheck_prompt = load_prompt("doctor_recheck")
        self.check_imaging_prompt = load_prompt("check_imaging")

    def run(self, ground_truth: Dict, case_id: Optional[str] = None) -> Dict:
        print("\n" + "=" * 60)
        print("Start medical diagnosis orchestration (MCTS)")
        print("=" * 60)

        self._initialize_memory(ground_truth, case_id=case_id)
        self._ensure_hypothesis_format()
        self._update_top_confidence()
        set_case_llm_phase("base")
        mcts_config = MCTSConfig(
            max_rollout_calls_per_search=self.mcts_rollout_call_cap_per_search
        )
        self.memory.update(
            {
                "mcts_step": 0,
                "mcts_trajectory": [],
                "mcts_eta": mcts_config.eta,
                "workflow_action_history": [],
                "backtrack_events": [],
                "mcts_snapshots": [],
                "active_snapshot_id": None,
                "missing_evidence": [],
                "potential_missing_evidence": [],
                "experience_knowledge": [],
                "retrieved_guidelines": [],
                "retrieved_cdc_cases": [],
                "evidence_importance": {},
                "experience_memory_available": self.experience_memory.available,
                "terminated_by": None,
                "finalization_used": False,
            }
        )

        # The paper's search starts from a current working-memory snapshot and
        # defines verification/backtracking after hypothesis generation. Seed
        # that Perceive/Hypothesize state with benchmark Tasks 1-2 so eta=4 can
        # still orchestrate retrieval/correction plus Tasks 3-5.
        if self.warm_start:
            self._warm_start_perceive_hypothesize(ground_truth)

        mcts = MCTSSearch(
            planning_llm=self._call_mcts_planning_llm,
            simulation_llm=self._call_rollout_simulation_llm,
            config=mcts_config,
            action_space=ACTION_SPACE,
        )
        rollout_calls_used_total = 0
        search_iteration = 0

        while int(self.memory.get("mcts_step", 0) or 0) < mcts.config.eta:
            if (
                self.mcts_rollout_call_cap_total > 0
                and rollout_calls_used_total >= self.mcts_rollout_call_cap_total
            ):
                print(
                    f"[MCTS] rollout call cap reached ({self.mcts_rollout_call_cap_total})"
                    f"stop search early at step={self.memory.get('mcts_step', 0)}"
                )
                self.memory.update({"terminated_by": "rollout_budget"})
                break
            state = self.memory.get_all()
            if state.get("hypothesis_illness"):
                self.memory.update({"missing_evidence": self._detect_missing_evidence()})
                state = self.memory.get_all()
            self._append_mcts_snapshot(reason="search_root_%s" % search_iteration)

            state = self.memory.get_all()
            per_search_cap = self.mcts_rollout_call_cap_per_search
            remaining_total = None
            if self.mcts_rollout_call_cap_total > 0:
                remaining_total = max(
                    0,
                    self.mcts_rollout_call_cap_total - rollout_calls_used_total,
                )
                effective_cap = (
                    min(per_search_cap, remaining_total)
                    if per_search_cap > 0
                    else remaining_total
                )
            else:
                effective_cap = per_search_cap
            mcts.config.max_rollout_calls_per_search = effective_cap

            valid_actions = mcts._filter_actions(state)
            candidate_count = min(mcts.config.K, len(valid_actions))
            if effective_cap > 0 and candidate_count > 1:
                fair_minimum = mcts._minimum_rollout_call_budget(candidate_count, state)
                total_is_only_constraint = (
                    remaining_total is not None
                    and remaining_total < fair_minimum
                    and (per_search_cap == 0 or per_search_cap >= fair_minimum)
                )
                if total_is_only_constraint:
                    print(
                        "[MCTS] remaining total rollout budget cannot fund a "
                        "complete Top-K x N search; stopping without biased partial rollouts"
                    )
                    self.memory.update({"terminated_by": "rollout_budget"})
                    break

            best_action = mcts.search(state)
            rollout_calls_used_total += int(mcts.last_search_stats.get("rollout_calls_used", 0) or 0)
            self._log_mcts_search(search_iteration, mcts.last_search_stats)

            if best_action == "a_term":
                terminal_state = record_action(
                    self.memory.get_all(), "a_term", ACTION_SPACE,
                    source="mcts_selected"
                )
                terminal_state["terminated_by"] = "a_term"
                self.memory.set_all(terminal_state)
                break

            if best_action == "a_back":
                missing = self._merge_missing_evidence(
                    state.get("missing_evidence", []) or self._detect_missing_evidence(),
                )
                self.memory.update({"missing_evidence": missing})
                did_backtrack, _bt_action = self._execute_backtrack(missing, ground_truth)
                if not did_backtrack:
                    self.memory.update({"terminated_by": "backtrack_unavailable"})
                    break
                self.memory.update({"missing_evidence": self._detect_missing_evidence()})
            else:
                set_case_llm_phase("base")
                self._execute_mcts_action(
                    best_action, ground_truth, count_mcts_action=True,
                    action_source="mcts_selected"
                )

                missing = self._detect_missing_evidence()
                self.memory.update({"missing_evidence": missing})

                if self._should_backtrack_after_action(best_action, missing):
                    did_backtrack, _bt_action = self._execute_backtrack(missing, ground_truth)
                    if did_backtrack:
                        self.memory.update({"missing_evidence": self._detect_missing_evidence()})

            self._ensure_hypothesis_format()
            self._update_top_confidence()

            latest_state = self.memory.get_all()
            if latest_state.get("diagnosis_result") and latest_state.get("treatment_plan"):
                self.memory.update({"terminated_by": "clinical_closure"})
                break
            search_iteration += 1

        if not self.memory.get("terminated_by"):
            self.memory.update({"terminated_by": "eta"})

        if self.force_finalization:
            self._finalize_current_diagnosis(ground_truth)
            self.memory.update({"finalization_used": True})

        recheck_summary = self.memory.get_recheck_summary()
        print("[log omitted: encoding-fixed]")
        print("[orchestrator] status update")

        print("\n" + "=" * 60)
        print("[log omitted: encoding-fixed]")
        print("=" * 60)
        return self.memory.get_all()

    def _append_mcts_snapshot(self, reason: str, workflow_stage: Optional[str] = None):
        state = append_snapshot(
            self.memory.get_all(), reason=reason, workflow_stage=workflow_stage
        )
        self.memory.update({"mcts_snapshots": state["mcts_snapshots"]})

    def _warm_start_perceive_hypothesize(self, ground_truth: Dict[str, Any]):
        """Seed E/H before MCTS without consuming the eta search horizon."""
        self._append_mcts_snapshot("warm_start_initial", "initial")
        self._execute_mcts_action(
            "task1_referral", ground_truth, count_mcts_action=False,
            action_source="warm_start"
        )

        self._append_mcts_snapshot("warm_start_before_doctor", "task1_referral")
        self._execute_mcts_action(
            "task2_doctor", ground_truth, count_mcts_action=False,
            action_source="warm_start"
        )

        missing = self._detect_missing_evidence()
        self.memory.update({"missing_evidence": missing})
        self._append_mcts_snapshot("warm_start_ready", "task2_doctor")
        if self._should_backtrack_after_action("task2_doctor", missing):
            self._execute_backtrack(missing, ground_truth)

    def _call_mcts_planning_llm(self, system_role: str, user_prompt: str) -> str:
        return call_llm_api_with_config(
            system_role,
            user_prompt,
            api_config=self.mcts_planning_config,
            api_tag="mcts_planning_llm",
        )

    def _call_rollout_simulation_llm(self, system_role: str, user_prompt: str) -> str:
        return call_llm_api_with_config(
            system_role,
            user_prompt,
            api_config=self.mcts_rollout_config,
            api_tag="mcts_rollout_simulation_llm",
        )

    @staticmethod
    def _assert_distinct_rollout_model(
        professional_config: Dict[str, str],
        rollout_config: Dict[str, str],
    ) -> None:
        def normalize_model_id(config: Dict[str, str]) -> str:
            model_id = str(config.get("model", "")).strip().lower()
            # Provider-qualified identifiers such as ``openai/gpt-x`` and
            # ``openai:gpt-x`` still refer to the same underlying model name.
            return model_id.rsplit("/", 1)[-1].rsplit(":", 1)[-1]

        professional_model = normalize_model_id(professional_config)
        rollout_model = normalize_model_id(rollout_config)
        if not rollout_model:
            raise ValueError(
                "MCTS rollout simulation requires an explicit model in "
                "MCTS_ROLLOUT_CONFIG (or MCTS_ROLLOUT_MODEL)."
            )
        if professional_model and rollout_model == professional_model:
            raise ValueError(
                "MCTS rollout simulation model must differ from the professional-agent "
                f"model; both resolve to {rollout_config.get('model')!r}. "
                "Set MCTS_ROLLOUT_MODEL to an independent simulation model."
            )

    def _log_mcts_search(self, step_idx: int, stats: Dict[str, Any]):
        print(f"[MCTS] step={step_idx + 1} chosen_action={stats.get('chosen_action')}")
        for child in stats.get("children", [])[:8]:
            print(
                f"  - action={child.get('action')} visit={child.get('visit_count')} "
                f"q={child.get('q_value')} prior={child.get('prior')}"
            )

    def _execute_professional_action_on_state(
        self,
        state: Dict[str, Any],
        action: str,
        ground_truth: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Execute one selected action with its professional agent.

        This path is used only after MCTS has selected an action. Rollout
        simulation lives entirely in ``MCTSSearch`` and cannot call this method.
        The function computes against a copy so the result can be committed
        atomically to ``StateMemory``.
        """
        sim_state = deepcopy(state)
        status = dict(sim_state.get("task_status", {}) or {})
        case_id = sim_state.get("case_id") or "unknown"

        if action == "task1_referral":
            l1_result = self.referral_agent.run_l1(
                sim_state.get("chief_complaint", ""),
                sim_state.get("present_illness", ""),
                case_id=case_id,
            )
            dept_l1 = l1_result.get("dept_l1")
            l2_result = self.referral_agent.run_l2(
                sim_state.get("chief_complaint", ""),
                sim_state.get("present_illness", ""),
                dept_l1,
                case_id=case_id,
            )
            sim_state["dept_l1"] = dept_l1
            sim_state["dept_l2"] = self._normalize_dept_l2(l2_result.get("dept_l2"))
            status[action] = "completed"

        elif action == "task2_doctor":
            result = self.doctor_agent.run(
                personal_info=sim_state.get("personal_info", {}),
                chief_complaint=sim_state.get("chief_complaint", ""),
                present_illness=sim_state.get("present_illness", ""),
                dept_l1=sim_state.get("dept_l1"),
                past_history=sim_state.get("past_history", ""),
                case_id=case_id,
                completed_physical_exams=sim_state.get("completed_physical_exams", []),
                completed_auxiliary_exams=sim_state.get("completed_auxiliary_exams", []),
            )
            hypotheses = self._normalize_hypothesis_items(result.get("hypothesis_illness", []))
            if hypotheses:
                sim_state["hypothesis_illness"] = hypotheses
            physical_exams = list(result.get("physical_exams", []) or [])
            auxiliary_exams = list(result.get("auxiliary_exams", []) or [])
            sim_state["predicted_physical_exams"] = physical_exams
            sim_state["predicted_auxiliary_exams"] = auxiliary_exams
            completed_aux = set(sim_state.get("completed_auxiliary_exams", []) or [])
            sim_state["pending_auxiliary_exams"] = [
                exam for exam in auxiliary_exams if exam not in completed_aux
            ]

            matched = self.doctor_agent.get_physical_exam_from_gt(
                ground_truth.get("physical_exam", {}), physical_exams
            )
            physical_results = dict(sim_state.get("physical_exams", {}) or {})
            completed_physical = list(sim_state.get("completed_physical_exams", []) or [])
            for predicted_exam in physical_exams:
                for gt_name, gt_value in matched.items():
                    if (
                        predicted_exam.lower() in gt_name.lower()
                        or gt_name.lower() in predicted_exam.lower()
                    ):
                        physical_results[predicted_exam] = gt_value
                        if predicted_exam not in completed_physical:
                            completed_physical.append(predicted_exam)
                        break
            sim_state["physical_exams"] = physical_results
            sim_state["completed_physical_exams"] = completed_physical
            status[action] = "completed"

        elif action == "task3_imaging":
            pending = list(sim_state.get("pending_auxiliary_exams", []) or [])
            result = self.imaging_agent.run(
                auxiliary_exams=pending,
                chief_complaint=sim_state.get("chief_complaint", ""),
                images_info=ground_truth.get("images_info", []),
                gt_img_paths=ground_truth.get("gt_img_paths", []),
                gt_auxiliary_exam=ground_truth.get("auxiliary_exam", {}),
                case_id=case_id,
                completed_imaging_exams=sim_state.get("completed_imaging_exams", []),
            )
            imaging_results = dict(sim_state.get("imaging_results", {}) or {})
            auxiliary_results = dict(sim_state.get("auxiliary_exams", {}) or {})
            completed_imaging = list(sim_state.get("completed_imaging_exams", []) or [])
            completed_auxiliary = list(sim_state.get("completed_auxiliary_exams", []) or [])
            completed_now: List[str] = []
            for exam_name, report in (result.get("imaging_results", {}) or {}).items():
                imaging_results[exam_name] = report
                auxiliary_results[exam_name] = report
                completed_now.append(exam_name)
                if exam_name not in completed_imaging:
                    completed_imaging.append(exam_name)
                if exam_name not in completed_auxiliary:
                    completed_auxiliary.append(exam_name)
            for exam_name, report in (result.get("non_imaging_results", {}) or {}).items():
                auxiliary_results[exam_name] = report
                completed_now.append(exam_name)
                if exam_name not in completed_auxiliary:
                    completed_auxiliary.append(exam_name)
            sim_state["imaging_results"] = imaging_results
            sim_state["auxiliary_exams"] = auxiliary_results
            sim_state["completed_imaging_exams"] = completed_imaging
            sim_state["completed_auxiliary_exams"] = completed_auxiliary
            sim_state["pending_auxiliary_exams"] = [
                exam for exam in pending if exam not in set(completed_now + completed_auxiliary)
            ]
            status[action] = "completed"

        elif action == "task4_diagnosis":
            result = self.diagnosis_agent.run(
                personal_info=sim_state.get("personal_info", {}),
                chief_complaint=sim_state.get("chief_complaint", ""),
                present_illness=sim_state.get("present_illness", ""),
                dept_l1=sim_state.get("dept_l1"),
                past_history=sim_state.get("past_history", ""),
                examination_results=self._reasoning_context_from_state(sim_state),
                hypothesis_illness=sim_state.get("hypothesis_illness", []),
                case_id=case_id,
            )
            updated = self._normalize_hypothesis_items(result.get("updated_hypothesis", []))
            if updated:
                sim_state["hypothesis_illness"] = updated
            diagnosis = result.get("diagnosis_result", []) or []
            if not diagnosis and updated:
                diagnosis = [max(updated, key=lambda item: item.get("confidence", 0.0))["disease"]]
            sim_state["diagnosis_result"] = diagnosis
            sim_state["agent_missing_evidence"] = result.get("missing_evidence", [])
            status[action] = "completed" if diagnosis else "need_recheck"

        elif action == "task5_treatment":
            result = self.treatment_agent.run(
                personal_info=sim_state.get("personal_info", {}),
                chief_complaint=sim_state.get("chief_complaint", ""),
                present_illness=sim_state.get("present_illness", ""),
                past_history=sim_state.get("past_history", ""),
                examination_results=self._reasoning_context_from_state(sim_state),
                department=sim_state.get("dept_l1"),
                diagnosis_result=sim_state.get("diagnosis_result", []),
                case_id=case_id,
            )
            sim_state["treatment_plan"] = result.get("treatment_plan", []) or []
            status[action] = "completed" if sim_state["treatment_plan"] else "pending"

        elif action == "a_rag":
            retrieval = self.experience_memory.retrieve(sim_state)
            sim_state.update(
                {
                    "experience_knowledge": retrieval.get("knowledge", []),
                    "potential_missing_evidence": retrieval.get("potential_missing_evidence", []),
                    "retrieved_guidelines": retrieval.get("retrieved_guidelines", []),
                    "retrieved_cdc_cases": retrieval.get("retrieved_cdc_cases", []),
                    "evidence_importance": retrieval.get("evidence_importance", {}),
                    "experience_retrieval_backend": retrieval.get("retrieval_backend"),
                }
            )

        sim_state["task_status"] = status
        normalized = self._normalize_hypothesis_items(sim_state.get("hypothesis_illness", []))
        sim_state["top_hypothesis_confidence"] = max(
            (float(item.get("confidence", 0.0)) for item in normalized),
            default=0.0,
        )
        return sim_state

    def _reasoning_context_from_state(self, state: Dict[str, Any]) -> str:
        parts: List[str] = []
        for label, field in (
            ("Physical Exam", "physical_exams"),
            ("Auxiliary Exam", "auxiliary_exams"),
            ("Imaging Result", "imaging_results"),
        ):
            values = state.get(field, {}) or {}
            if values:
                parts.append(f"[{label}]\n" + "\n".join(f"{k}: {v}" for k, v in values.items()))
        if state.get("experience_knowledge"):
            parts.append("[Experience Memory K_t]\n" + "\n".join(
                str(item) for item in state.get("experience_knowledge", [])
            ))
        if state.get("potential_missing_evidence"):
            parts.append("[Potential Missing Evidence E_t^p]\n" + "\n".join(
                str(item) for item in state.get("potential_missing_evidence", [])
            ))
        return "\n\n".join(parts) if parts else "No exam results available"

    def _decide_backtrack_from_state(
        self,
        state: Dict[str, Any],
        missing_evidence: List[str],
    ) -> Dict[str, Any]:
        workflow_history = list(state.get("workflow_action_history", []) or [])
        before_index = len(workflow_history) - 1 if workflow_history else None
        prompt = BACKTRACK_USER.format(
            hypothesis_illness=state.get("hypothesis_illness", []),
            missing_evidence=missing_evidence,
            mcts_trajectory=state.get("mcts_trajectory", []),
            mcts_step=state.get("mcts_step", 0),
            snapshot_metadata=snapshot_prompt_metadata(
                state, before_workflow_action_index=before_index
            ),
        )
        response = call_llm_api(BACKTRACK_SYSTEM, prompt)
        result = parse_json_from_response(response)
        if not isinstance(result, dict):
            return {}
        return result

    def _execute_mcts_action(
        self, action: str, ground_truth: Dict,
        count_mcts_action: bool = True,
        action_source: str = "mcts_selected",
    ):
        if action in ACTION_SPACE and action not in ("a_back", "a_term"):
            current_state = self.memory.get_all()
            transitioned = self._execute_professional_action_on_state(
                current_state,
                action,
                ground_truth,
            )
            transitioned = record_action(
                transitioned, action, ACTION_SPACE, source=action_source,
                count_mcts=count_mcts_action
            )
            self.memory.set_all(transitioned)

        self._ensure_hypothesis_format()
        self._update_top_confidence()

    def _run_rag_retrieval(self):
        state = self.memory.get_all()
        retrieval = self.experience_memory.retrieve(state)
        self.memory.update(
            {
                "experience_knowledge": retrieval.get("knowledge", []),
                "potential_missing_evidence": retrieval.get("potential_missing_evidence", []),
                "retrieved_guidelines": retrieval.get("retrieved_guidelines", []),
                "retrieved_cdc_cases": retrieval.get("retrieved_cdc_cases", []),
                "evidence_importance": retrieval.get("evidence_importance", {}),
                "experience_retrieval_backend": retrieval.get("retrieval_backend"),
            }
        )
        print("[RAG] retrieved experience memory")

    def _detect_missing_evidence(self) -> List[str]:
        state = self.memory.get_all()
        prompt = MISSING_EVIDENCE_USER.format(
            chief_complaint=state.get("chief_complaint", ""),
            physical_exams=state.get("physical_exams", {}),
            auxiliary_exams=state.get("auxiliary_exams", {}),
            imaging_results=state.get("imaging_results", {}),
            past_history=state.get("past_history", ""),
            hypothesis_illness=state.get("hypothesis_illness", []),
            experience_knowledge=state.get("experience_knowledge", []),
            potential_missing_evidence=state.get("potential_missing_evidence", []),
        )
        response = call_llm_api(MISSING_EVIDENCE_SYSTEM, prompt)
        result = parse_json_from_response(response)
        if isinstance(result, list):
            detected = [str(item).strip() for item in result if str(item).strip()]
        else:
            raise RuntimeError("Missing-evidence verifier returned invalid JSON")
        return self._merge_missing_evidence(detected)

    def _merge_missing_evidence(self, *groups: Any) -> List[Any]:
        merged: List[Any] = []
        seen = set()
        for group in groups:
            if not group:
                continue
            items = group if isinstance(group, list) else [group]
            for item in items:
                key = str(item).strip()
                if not key or key in seen:
                    continue
                seen.add(key)
                merged.append(item)
        return merged

    def _get_reasoning_context(self) -> str:
        state = self.memory.get_all()
        parts = [self.memory.get_examination_summary()]
        knowledge = state.get("experience_knowledge", [])
        potential = state.get("potential_missing_evidence", [])
        if knowledge:
            parts.append("[Experience Memory K_t]\n" + "\n".join(str(item) for item in knowledge))
        if potential:
            parts.append("[Potential Missing Evidence E_t^p]\n" + "\n".join(str(item) for item in potential))
        return "\n\n".join(part for part in parts if part)

    def _decide_backtrack(self, missing_evidence: List[str]) -> Dict[str, Any]:
        state = self.memory.get_all()
        return self._decide_backtrack_from_state(state, missing_evidence)

    def _has_recheck_budget(self) -> bool:
        state = self.memory.get_all()
        return int(state.get("recheck_total", 0) or 0) < int(state.get("max_recheck", 0) or 0)

    def _has_mcts_action_budget(self, required_actions: int = 1) -> bool:
        state = self.memory.get_all()
        current_step = int(state.get("mcts_step", 0) or 0)
        eta = int(state.get("mcts_eta", 0) or 0)
        return eta <= 0 or current_step + max(0, int(required_actions)) <= eta

    def _should_backtrack_after_action(self, action: str, missing_evidence: List[str]) -> bool:
        if not missing_evidence:
            return False
        if not self._has_recheck_budget():
            return False
        if not self._has_mcts_action_budget(required_actions=2):
            return False

        # Eq. (8): after hypothesis generation, any verified critical evidence
        # gap triggers the corrective backtracking process.
        return action in ("task2_doctor", "task4_diagnosis")

    def _execute_backtrack(
        self,
        missing_evidence: List[str],
        ground_truth: Dict,
    ) -> Tuple[bool, Optional[str]]:
        if not missing_evidence:
            return False, None
        if not self._has_recheck_budget():
            print("[log omitted: encoding-fixed]")
            return False, None

        if not self._has_mcts_action_budget(required_actions=2):
            print("[Backtrack] eta has no room for a_back plus its target action")
            return False, None

        latest_state = self.memory.get_all()
        workflow_history = list(latest_state.get("workflow_action_history", []) or [])
        before_index = len(workflow_history) - 1 if workflow_history else None
        with llm_phase("recheck"):
            decision = self._decide_backtrack(missing_evidence)
        target_snapshot = resolve_backtrack_snapshot(
            latest_state, decision, before_workflow_action_index=before_index
        )
        if target_snapshot is None:
            print("[log omitted: encoding-fixed]")
            return False, None

        preview_recovered = recover_snapshot_state(latest_state, target_snapshot)
        requested_action = decision.get("target_action")
        target_action = choose_valid_backtrack_action(
            preview_recovered, requested_action, ACTION_SPACE,
            eta=int(latest_state.get("mcts_eta", 0) or 0)
        )
        if target_action is None:
            print("[Backtrack] no valid target action after snapshot restoration")
            return False, None
        if not self.memory.increment_recheck():
            print("[log omitted: encoding-fixed]")
            return False, None

        state_after_inc = self.memory.get_all()
        back_state = record_action(
            state_after_inc, "a_back", ACTION_SPACE,
            source="committed_backtrack"
        )
        recovered = recover_snapshot_state(back_state, target_snapshot)
        recovered = append_backtrack_event(
            recovered,
            target_snapshot,
            requested_action=requested_action,
            executed_action=target_action,
            source="committed",
        )
        recovered = append_snapshot(
            recovered, reason="committed_backtrack_restored",
            workflow_stage=target_snapshot["workflow_stage"]
        )
        self.memory.set_all(recovered)

        with llm_phase("recheck"):
            self._execute_mcts_action(
                target_action, ground_truth, count_mcts_action=True,
                action_source="committed_backtrack_target"
            )
        return True, target_action

    def _update_top_confidence(self):
        state = self.memory.get_all()
        hypothesis = self._normalize_hypothesis_items(state.get("hypothesis_illness", []))
        top_conf = max((float(item.get("confidence", 0.0)) for item in hypothesis), default=0.0)
        self.memory.update({"top_hypothesis_confidence": top_conf})

    def _ensure_hypothesis_format(self):
        state = self.memory.get_all()
        raw_hypothesis = state.get("hypothesis_illness", [])
        normalized = self._normalize_hypothesis_items(raw_hypothesis)
        if not normalized and isinstance(raw_hypothesis, list) and raw_hypothesis:
            names = [str(item).strip() for item in raw_hypothesis if str(item).strip()]
            normalized = self._infer_hypothesis_confidence_with_llm(names)
        if normalized:
            self.memory.set_hypothesis_illness(normalized)

    def _normalize_hypothesis_items(self, hypothesis: Any) -> List[Dict[str, Any]]:
        if not isinstance(hypothesis, list):
            return []

        normalized: List[Dict[str, Any]] = []
        for item in hypothesis:
            if isinstance(item, str):
                name = item.strip()
                if not name:
                    continue
                normalized.append({"name": name, "disease": name, "evidence": [], "confidence": 0.5})
                continue

            if not isinstance(item, dict):
                continue

            name = str(item.get("name") or item.get("disease") or "").strip()
            if not name:
                continue

            evidence = item.get("evidence", [])
            if isinstance(evidence, str):
                evidence = [evidence]
            if not isinstance(evidence, list):
                evidence = []

            try:
                confidence = float(item.get("confidence", 0.0))
            except (TypeError, ValueError):
                confidence = 0.0
            confidence = max(0.0, min(1.0, confidence))

            normalized.append(
                {
                    "name": name,
                    "disease": name,
                    "evidence": [str(e).strip() for e in evidence if str(e).strip()],
                    "confidence": confidence,
                }
            )
        return normalized

    def _infer_hypothesis_confidence_with_llm(self, names: List[str]) -> List[Dict[str, Any]]:
        if not names:
            return []

        state = self.memory.get_all()
        prompt = HYPOTHESIS_CONFIDENCE_USER.format(
            personal_info=state.get("personal_info", {}),
            chief_complaint=state.get("chief_complaint", ""),
            present_illness=state.get("present_illness", ""),
            physical_exams=state.get("physical_exams", {}),
            auxiliary_exams=state.get("auxiliary_exams", {}),
            imaging_results=state.get("imaging_results", {}),
            hypothesis_names=names,
        )
        response = call_llm_api(HYPOTHESIS_CONFIDENCE_SYSTEM, prompt)
        result = parse_json_from_response(response)
        if isinstance(result, list):
            normalized = self._normalize_hypothesis_items(result)
            if normalized:
                return normalized

        fallback: List[Dict[str, Any]] = []
        for idx, name in enumerate(names):
            fallback.append(
                {
                    "name": name,
                    "disease": name,
                    "evidence": [],
                    "confidence": max(0.1, 0.7 - 0.1 * idx),
                }
            )
        return fallback

    def _finalize_current_diagnosis(self, ground_truth: Dict):
        state = self.memory.get_all()
        if not state.get("dept_l1"):
            self._execute_task1_referral()

        state = self.memory.get_all()
        if not state.get("hypothesis_illness"):
            self._execute_task2_doctor(ground_truth)

        state = self.memory.get_all()
        has_exam = bool(state.get("physical_exams")) or bool(state.get("auxiliary_exams")) or bool(
            state.get("imaging_results")
        )
        if not has_exam and state.get("pending_auxiliary_exams"):
            self._execute_task3_imaging(ground_truth)

        state = self.memory.get_all()
        if not state.get("diagnosis_result"):
            self._execute_task4_diagnosis(ground_truth, allow_internal_recheck=False)

        state = self.memory.get_all()
        if not state.get("treatment_plan"):
            self._execute_task5_treatment()

    def _initialize_memory(self, ground_truth: Dict, case_id: Optional[str] = None):
        """Initialize working memory for a new case."""
        print("[orchestrator] status update")

        self.memory.initialize(
            personal_info=ground_truth.get("personal_info", {}),
            chief_complaint=ground_truth.get("zhusu", ""),
            present_illness=ground_truth.get("xianbingshi", ""),
            case_id=str(case_id) if case_id is not None else None
        )
        self.memory.update({"max_recheck": self.max_recheck, "recheck_total": 0})

        self.memory.set_past_history(ground_truth.get("jiwangshi", ""))

        print("[orchestrator] status update")
        print("[orchestrator] status update")
        print("[orchestrator] status update")
        print("[orchestrator] status update")

    def _execute_task1_referral(self):
        """Execute referral and department selection."""
        print("[orchestrator] status update")

        state = self.memory.get_all()
        chief_complaint = state["chief_complaint"]
        personal_info = state["personal_info"]
        present_illness = state["present_illness"]
        case_id = state.get("case_id") or "unknown"

        referral_l1_result = self.referral_agent.run_l1(
            chief_complaint,
            present_illness,
            case_id=case_id
        )
        dept_l1 = referral_l1_result["dept_l1"]

        print("[orchestrator] status update")

        verify_l1_result = self._verify_departments_with_llm(
            personal_info=personal_info,
            chief_complaint=chief_complaint,
            present_illness=present_illness,
            past_history=None,
            dept_l1=dept_l1,
            dept_l2=[],
            case_id=case_id
        )

        dept_l1_correct = verify_l1_result.get("dept_l1_correct", True)

        if dept_l1_correct:
            final_dept_l1 = dept_l1
            print("[log omitted: encoding-fixed]")
        else:
            print("[orchestrator] status update")
            print("[orchestrator] status update")

            final_l1_result = self._finalize_departments_with_llm(
                personal_info=personal_info,
                chief_complaint=chief_complaint,
                present_illness=present_illness,
                past_history=None,
                agent_output=referral_l1_result,
                orchestrator_output=verify_l1_result,
                case_id=case_id
            )

            final_dept_l1 = final_l1_result.get("final_dept_l1") or verify_l1_result.get("suggested_dept_l1") or dept_l1
            print("[orchestrator] status update")
            print("[orchestrator] status update")

        referral_l2_result = self.referral_agent.run_l2(
            chief_complaint,
            present_illness,
            final_dept_l1,
            case_id=case_id
        )
        dept_l2 = referral_l2_result["dept_l2"]

        print("[orchestrator] status update")

        verify_l2_result = self._verify_departments_with_llm(
            personal_info=personal_info,
            chief_complaint=chief_complaint,
            present_illness=present_illness,
            past_history=None,
            dept_l1=final_dept_l1,
            dept_l2=dept_l2,
            case_id=case_id
        )

        dept_l1_correct = verify_l2_result.get("dept_l1_correct", True)
        dept_l2_correct = verify_l2_result.get("dept_l2_correct", True)

        if dept_l1_correct and dept_l2_correct:
            final_dept_l2 = dept_l2
            print("[log omitted: encoding-fixed]")
        else:
            print("[orchestrator] status update")
            print("[orchestrator] status update")
            print(
                "  - Suggested departments: "
                f"{verify_l2_result.get('suggested_dept_l1', '')}, "
                f"{verify_l2_result.get('suggested_dept_l2', '')}"
            )

            final_result = self._finalize_departments_with_llm(
                personal_info=personal_info,
                chief_complaint=chief_complaint,
                present_illness=present_illness,
                past_history=None,
                agent_output=referral_l2_result,
                orchestrator_output=verify_l2_result,
                case_id=case_id
            )

            final_dept_l1 = final_result.get("final_dept_l1") or verify_l2_result.get("suggested_dept_l1") or final_dept_l1
            final_dept_l2 = self._normalize_dept_l2(
                final_result.get("final_dept_l2")
                or verify_l2_result.get("suggested_dept_l2")
                or dept_l2
            )

            print("[orchestrator] status update")
            print("[orchestrator] status update")

        # Orchestration processing step.
        self.memory.set_department(final_dept_l1, final_dept_l2)
        print("[log omitted: encoding-fixed]")

    def _execute_task2_doctor(self, ground_truth: Dict, is_recheck: bool = False):
        """Execute doctor hypothesis and exam planning."""
        print("[orchestrator] status update")

        state = self.memory.get_all()

        completed_physical = state.get("completed_physical_exams", []) if is_recheck else None
        completed_auxiliary = state.get("completed_auxiliary_exams", []) if is_recheck else None

        with llm_phase("recheck" if is_recheck else "base"):
            doctor_result = self.doctor_agent.run(
                personal_info=state["personal_info"],
                chief_complaint=state["chief_complaint"],
                present_illness=state["present_illness"],
                dept_l1=state["dept_l1"],
                # dept_l2=state["dept_l2"],
                past_history=state["past_history"],
                case_id=state.get("case_id") or "unknown",
                completed_physical_exams=completed_physical,
                completed_auxiliary_exams=completed_auxiliary
            )

        hypothesis = self._normalize_hypothesis_items(doctor_result.get("hypothesis_illness", []))
        physical_exams = doctor_result["physical_exams"]
        auxiliary_exams = doctor_result["auxiliary_exams"]

        print("[orchestrator] status update")
        print("[orchestrator] status update")
        print("[orchestrator] status update")

        dept_review = self._verify_departments_with_llm(
            personal_info=state["personal_info"],
            chief_complaint=state["chief_complaint"],
            present_illness=state["present_illness"],
            past_history=state["past_history"],
            dept_l1=state["dept_l1"],
            dept_l2=state["dept_l2"],
            case_id=state.get("case_id") or "unknown"
        )

        dept_l1_correct = dept_review.get("dept_l1_correct", True)
        dept_l2_correct = dept_review.get("dept_l2_correct", True)

        if dept_l1_correct and dept_l2_correct:
            final_dept_l1 = state["dept_l1"]
            final_dept_l2 = state["dept_l2"]
            print("[log omitted: encoding-fixed]")
        else:
            print("[orchestrator] status update")
            print("[orchestrator] status update")
            print(
                "  - Suggested departments: "
                f"{dept_review.get('suggested_dept_l1', '')}, "
                f"{dept_review.get('suggested_dept_l2', '')}"
            )
            final_result = self._finalize_departments_with_llm(
                personal_info=state["personal_info"],
                chief_complaint=state["chief_complaint"],
                present_illness=state["present_illness"],
                past_history=state["past_history"],
                agent_output=doctor_result,
                orchestrator_output=dept_review,
                case_id=state.get("case_id") or "unknown"
            )
            final_dept_l1 = final_result.get("final_dept_l1") or dept_review.get("suggested_dept_l1") or state["dept_l1"]
            final_dept_l2 = self._normalize_dept_l2(
                final_result.get("final_dept_l2")
                or dept_review.get("suggested_dept_l2")
                or state["dept_l2"]
            )
            print("[orchestrator] status update")

        self.memory.update_department(final_dept_l1, final_dept_l2)

        if not is_recheck:
            self.memory.set_hypothesis_illness(hypothesis)
        else:
            existing_hypothesis = self._normalize_hypothesis_items(state.get("hypothesis_illness", []))
            existing_diseases = {h.get("disease") for h in existing_hypothesis}
            for h in hypothesis:
                if h.get("disease") not in existing_diseases:
                    existing_hypothesis.append(h)
            self.memory.set_hypothesis_illness(existing_hypothesis)

        # gt_physical = ground_truth.get("physical_exam", {})
        # matched_physical = self.doctor_agent.get_physical_exam_from_gt(gt_physical, physical_exams)
        #
        # for pred_exam in physical_exams:
        #     exam_result = None
        #     for gt_key, gt_value in matched_physical.items():
        #         if pred_exam.lower() in gt_key.lower() or gt_key.lower() in pred_exam.lower():
        #             exam_result = gt_value
        #             break
        #     if exam_result:
        #         self.memory.add_physical_exam(pred_exam, exam_result)
        print("[log omitted: encoding-fixed]")
        self._record_physical_exams_from_gt(ground_truth, physical_exams)

        if not is_recheck:
            self.memory.update({
                "predicted_physical_exams": physical_exams,
                "predicted_auxiliary_exams": auxiliary_exams
            })
        else:
            existing_physical = state.get("predicted_physical_exams", [])
            existing_auxiliary = state.get("predicted_auxiliary_exams", [])
            self.memory.update({
                "predicted_physical_exams": list(set(existing_physical + physical_exams)),
                "predicted_auxiliary_exams": list(set(existing_auxiliary + auxiliary_exams))
            })

        completed_aux = set(state.get("completed_auxiliary_exams", []))
        pending_auxiliary = [e for e in auxiliary_exams if e not in completed_aux]

        if is_recheck:
            existing_pending = state.get("pending_auxiliary_exams", [])
            combined_pending = list(set(existing_pending + pending_auxiliary) - completed_aux)
            self.memory.update({"pending_auxiliary_exams": combined_pending})
        else:
            self.memory.update({"pending_auxiliary_exams": pending_auxiliary})

        self.memory.mark_doctor_task_complete()
        self._ensure_hypothesis_format()
        self._update_top_confidence()
        print("[log omitted: encoding-fixed]")

    def _execute_task3_imaging(self, ground_truth: Dict, is_recheck: bool = False):
        """Execute imaging and auxiliary exam handling."""
        print("[orchestrator] status update")

        state = self.memory.get_all()
        auxiliary_exams = state.get("pending_auxiliary_exams", [])

        if not auxiliary_exams:
            print("[log omitted: encoding-fixed]")
            self.memory.mark_imaging_task_complete()
            return

        completed_imaging = state.get("completed_imaging_exams", []) if is_recheck else None

        with llm_phase("recheck" if is_recheck else "base"):
            imaging_result = self.imaging_agent.run(
                auxiliary_exams=auxiliary_exams,
                chief_complaint=state["chief_complaint"],
                images_info=ground_truth.get("images_info", []),
                gt_img_paths=ground_truth.get("gt_img_paths", []),
                gt_auxiliary_exam=ground_truth.get("auxiliary_exam", {}),
                case_id=state.get("case_id") or "unknown",
                completed_imaging_exams=completed_imaging
            )

        completed_exams = []
        imaging_inputs = imaging_result.get("imaging_inputs", {})
        for exam_type, result in imaging_result["imaging_results"].items():
            reviewed = self._review_imaging_report(
                exam_type=exam_type,
                report=result,
                image_paths=imaging_inputs.get(exam_type, {}).get("image_paths", []),
                case_id=state.get("case_id") or "unknown"
            )
            final_report = reviewed or result
            self.memory.add_imaging_result(exam_type, final_report)
            self.memory.add_auxiliary_exam(exam_type, final_report)
            completed_exams.append(exam_type)
            print("[log omitted: encoding-fixed]")

        for exam_type, result in imaging_result["non_imaging_results"].items():
            self.memory.add_auxiliary_exam(exam_type, result)
            completed_exams.append(exam_type)
            print("[log omitted: encoding-fixed]")

        current_pending = set(state.get("pending_auxiliary_exams", []))
        current_completed = set(state.get("completed_auxiliary_exams", []))
        remaining_pending = list(current_pending - current_completed - set(completed_exams))
        self.memory.update({"pending_auxiliary_exams": remaining_pending})

        self.memory.mark_imaging_task_complete()
        print("[log omitted: encoding-fixed]")

    def _execute_task4_diagnosis(self, ground_truth: Dict, allow_internal_recheck: bool = True):
        """Execute diagnosis synthesis."""
        print("[orchestrator] status update")

        state = self.memory.get_all()

        diagnosis_result = self.diagnosis_agent.run(
            personal_info=state["personal_info"],
            chief_complaint=state["chief_complaint"],
            present_illness=state["present_illness"],
            dept_l1=state["dept_l1"],
            # dept_l2=state["dept_l2"],
            past_history=state["past_history"],
            examination_results=self._get_reasoning_context(),
            hypothesis_illness=state.get("hypothesis_illness", []),
            case_id=state.get("case_id") or "unknown"
        )

        updated_hypothesis = self._normalize_hypothesis_items(diagnosis_result["updated_hypothesis"])
        confirmed_diagnosis = diagnosis_result["diagnosis_result"]
        missing_evidence = diagnosis_result["missing_evidence"]
        need_recheck = diagnosis_result["need_recheck"]

        print("[orchestrator] status update")
        print("[orchestrator] status update")
        print("[orchestrator] status update")

        self.memory.update_hypothesis_illness(updated_hypothesis)
        self._update_top_confidence()

        should_recheck = self._should_recheck_with_llm(diagnosis_result)

        if should_recheck and missing_evidence and allow_internal_recheck:
            with llm_phase("recheck"):
                print("[orchestrator] status update")

                completed_physical = set(state.get("completed_physical_exams", []))
                completed_auxiliary = set(state.get("completed_auxiliary_exams", []))

                suggested_exams = self._suggest_recheck_exams_with_llm(
                    missing_evidence=missing_evidence,
                    completed_physical=sorted(completed_physical),
                    completed_auxiliary=sorted(completed_auxiliary),
                    case_id=state.get("case_id") or "unknown"
                )
                if not suggested_exams.get("physical_exams") and not suggested_exams.get("auxiliary_exams"):
                    suggested_exams = self.diagnosis_agent.get_suggested_exams_from_missing(missing_evidence)

                new_physical = suggested_exams.get("physical_exams", [])
                new_auxiliary = suggested_exams.get("auxiliary_exams", [])

                gt_physical_keys = set(ground_truth.get("physical_exam", {}).keys())
                gt_auxiliary_keys = set(ground_truth.get("auxiliary_exam", {}).keys())

                valid_new_physical = []
                for exam in new_physical:
                    if exam in completed_physical:
                        print("[log omitted: encoding-fixed]")
                        continue
                    gt_match = self._find_gt_match(exam, gt_physical_keys)
                    if gt_match:
                        valid_new_physical.append(gt_match)
                        print("[log omitted: encoding-fixed]")
                    else:
                        print("[log omitted: encoding-fixed]")

                valid_new_auxiliary = []
                for exam in new_auxiliary:
                    if exam in completed_auxiliary:
                        print("[log omitted: encoding-fixed]")
                        continue
                    gt_match = self._find_gt_match(exam, gt_auxiliary_keys)
                    if gt_match:
                        valid_new_auxiliary.append(gt_match)
                        print("[log omitted: encoding-fixed]")
                    else:
                        print("[log omitted: encoding-fixed]")

                if valid_new_physical or valid_new_auxiliary:
                    # Bug fix:
                    # only increment once and do not perform bool arithmetic.
                    can_continue = self.memory.increment_recheck()

                    if can_continue:
                        existing_pred_physical = state.get("predicted_physical_exams", [])
                        existing_pred_auxiliary = state.get("predicted_auxiliary_exams", [])
                        self.memory.update({
                            "predicted_physical_exams": list(set(existing_pred_physical + valid_new_physical)),
                            "predicted_auxiliary_exams": list(set(existing_pred_auxiliary + valid_new_auxiliary))
                        })
                        if valid_new_physical:
                            self._record_physical_exams_from_gt(ground_truth, valid_new_physical)

                            current_pending = state.get("pending_auxiliary_exams", [])
                            combined = list(set(current_pending + valid_new_auxiliary))
                            self.memory.update({"pending_auxiliary_exams": combined})

                        if valid_new_auxiliary:
                            print("[orchestrator] status update")
                            self._execute_task3_imaging(ground_truth, is_recheck=True)

                        return self._execute_task4_diagnosis(ground_truth, allow_internal_recheck=True)
                    else:
                        print("[log omitted: encoding-fixed]")
                else:
                    print("[log omitted: encoding-fixed]")
        elif should_recheck and not allow_internal_recheck:
            self.memory.mark_diagnosis_need_recheck()

        if confirmed_diagnosis:
            self.memory.set_diagnosis_result(confirmed_diagnosis)
            print("[log omitted: encoding-fixed]")
        else:
            if updated_hypothesis:
                sorted_hypothesis = sorted(updated_hypothesis,
                                           key=lambda x: x.get("confidence", 0),
                                           reverse=True)
                top_disease = sorted_hypothesis[0].get("disease", "unknown")
                self.memory.set_diagnosis_result([top_disease])
                print("[orchestrator] status update")
            else:
                self.memory.set_diagnosis_result(["\u5f85\u8fdb\u4e00\u6b65\u8bca\u65ad"])

        print("[log omitted: encoding-fixed]")

    def _execute_task5_treatment(self):
        """Execute treatment planning."""
        print("[orchestrator] status update")

        state = self.memory.get_all()

        treatment_result = self.treatment_agent.run(
            personal_info=state["personal_info"],
            chief_complaint=state["chief_complaint"],
            present_illness=state["present_illness"],
            past_history=state["past_history"],
            examination_results=self._get_reasoning_context(),
            department=state["dept_l1"],
            diagnosis_result=state.get("diagnosis_result", []),
            case_id=state.get("case_id") or "unknown"
        )

        treatment_plan = treatment_result["treatment_plan"]
        reasoning = treatment_result["reasoning"]

        print("[orchestrator] status update")
        print("[orchestrator] status update")

        self.memory.set_treatment_plan(treatment_plan)
        print("[log omitted: encoding-fixed]")

    def _record_physical_exams_from_gt(self, ground_truth: Dict, physical_exams: List[str]):
        """Record available physical exam results from ground truth."""
        if not physical_exams:
            return

        gt_physical = ground_truth.get("physical_exam", {})
        matched_physical = self.doctor_agent.get_physical_exam_from_gt(gt_physical, physical_exams)

        for pred_exam in physical_exams:
            exam_result = None
            for gt_key, gt_value in matched_physical.items():
                if pred_exam.lower() in gt_key.lower() or gt_key.lower() in pred_exam.lower():
                    exam_result = gt_value
                    break
            if exam_result:
                self.memory.add_physical_exam(pred_exam, exam_result)
                print("[log omitted: encoding-fixed]")

    def _normalize_dept_l2(self, dept_l2: Any) -> List[str]:
        """Normalize secondary department labels."""
        if dept_l2 is None:
            return []
        if isinstance(dept_l2, list):
            return [d for d in dept_l2 if d]
        if isinstance(dept_l2, str):
            normalized = (
                dept_l2.replace("\uff0c", ",")
                .replace("\u3001", ",")
                .replace("\uff1b", ",")
                .replace(";", ",")
                .replace("\n", ",")
            )
            parts = [p.strip() for p in normalized.split(",") if p.strip()]
            return parts
        return []

    def _verify_departments_with_llm(
        self,
        personal_info: Dict,
        chief_complaint: str,
        present_illness: str,
        past_history: Optional[str],
        dept_l1: str,
        dept_l2: List[str],
        case_id: str
    ) -> Dict:
        """Ask the LLM to verify department predictions."""
        prompt = self.dept_verify_prompt.format(
            case_id=case_id,
            personal_info=str(personal_info),
            chief_complaint=chief_complaint,
            present_illness=present_illness,
            past_history=past_history if past_history else "",
            dept_l1=dept_l1,
            dept_l2=", ".join(self._normalize_dept_l2(dept_l2)) if dept_l2 else "",
        )
        response = call_llm_api("You are a medical orchestrator.", prompt)
        result = parse_json_from_response(response)
        if isinstance(result, dict):
            return result
        return {
            "dept_l1_correct": True,
            "dept_l2_correct": True,
            "reason": "",
            "suggested_dept_l1": None,
            "suggested_dept_l2": None
        }

    def _finalize_departments_with_llm(
        self,
        personal_info: Dict,
        chief_complaint: str,
        present_illness: str,
        past_history: Optional[str],
        agent_output: Dict,
        orchestrator_output: Dict,
        case_id: str
    ) -> Dict:
        """Ask the LLM to finalize department predictions."""
        prompt = self.dept_final_prompt.format(
            case_id=case_id,
            personal_info=str(personal_info),
            chief_complaint=chief_complaint,
            present_illness=present_illness,
            past_history=past_history if past_history else "",
            agent_output=str(agent_output),
            orchestrator_output=str(orchestrator_output)
        )
        response = call_llm_api("You are a medical orchestrator.", prompt)
        result = parse_json_from_response(response)
        if isinstance(result, dict):
            return result
        return {
            "final_dept_l1": None,
            "final_dept_l2": None,
            "reasoning": "",
        }

    def _review_imaging_report(
        self,
        exam_type: str,
        report: str,
        image_paths: List[str],
        case_id: str
    ) -> str:
        """Review an imaging report with the LLM."""
        if not report:
            return report
        prompt = self.check_imaging_prompt.format(
            case_id=case_id,
            imaging_type=exam_type,
            report=report
        )
        if image_paths:
            response = img_api(image_paths, prompt)
        else:
            response = call_llm_api("You are a medical orchestrator.", prompt)
        if response and response != "Error":
            return response
        return report

    def _suggest_recheck_exams_with_llm(
        self,
        missing_evidence: List[Dict],
        completed_physical: List[str],
        completed_auxiliary: List[str],
        case_id: str
    ) -> Dict[str, List[str]]:
        """Ask the LLM to suggest recheck exams."""
        prompt = self.doctor_recheck_prompt.format(
            case_id=case_id,
            missing_evidence=str(missing_evidence),
            completed_physical=", ".join(completed_physical) if completed_physical else "None",
            completed_auxiliary=", ".join(completed_auxiliary) if completed_auxiliary else "None",
            valid_physical=", ".join(VALID_PHYSICAL_EXAMS),
            valid_auxiliary=", ".join(VALID_AUXILIARY_EXAMS)
        )
        response = call_llm_api("You are a medical orchestrator.", prompt)
        result = parse_json_from_response(response)
        if not isinstance(result, dict):
            return {"physical_exams": [], "auxiliary_exams": []}

        physical = result.get("physical_exams", [])
        auxiliary = result.get("auxiliary_exams", [])

        physical = self.doctor_agent._validate_exams(physical, VALID_PHYSICAL_EXAMS)
        auxiliary = self.doctor_agent._validate_exams(auxiliary, VALID_AUXILIARY_EXAMS)

        physical = [e for e in physical if e not in completed_physical]
        auxiliary = [e for e in auxiliary if e not in completed_auxiliary]

        return {
            "physical_exams": physical,
            "auxiliary_exams": auxiliary
        }

    def get_final_state(self) -> Dict:
        """Return the final orchestration state."""
        return self.memory.get_all()

    def set_mcts_planning_config(self, planning_config: Dict[str, str]):
        """Update the model config used for MCTS action scoring."""
        self.mcts_planning_config = dict(planning_config)

    def set_mcts_rollout_config(self, rollout_config: Dict[str, str]):
        """Update the independent model used for counterfactual rollouts."""
        candidate = dict(rollout_config)
        self._assert_distinct_rollout_model(LLM_CONFIG, candidate)
        self.mcts_rollout_config = candidate

    def set_orchestration_limits(
        self,
        rollout_call_cap_per_search: Optional[int] = None,
        rollout_call_cap_total: Optional[int] = None,
        max_recheck: Optional[int] = None,
    ):
        """Dynamically set rollout and backtrack limits."""
        if rollout_call_cap_per_search is not None:
            self.mcts_rollout_call_cap_per_search = max(
                0, int(rollout_call_cap_per_search or 0)
            )
        if rollout_call_cap_total is not None:
            self.mcts_rollout_call_cap_total = max(0, int(rollout_call_cap_total or 0))
        if max_recheck is not None:
            self.max_recheck = max(0, int(max_recheck or 0))

    def _make_llm_decision(self) -> Dict:
        """Ask the LLM for an orchestration decision."""
        state = self.memory.get_all()

        task_status = state.get("task_status", {})
        completed_tasks = [k for k, v in task_status.items() if v == "completed"]

        completed_exams = (
            state.get("completed_physical_exams", []) +
            state.get("completed_auxiliary_exams", [])
        )
        hypothesis = self._normalize_hypothesis_items(state.get("hypothesis_illness", []))
        if hypothesis:
            max_confidence = max(h.get("confidence", 0) for h in hypothesis)
        else:
            max_confidence = "unknown"

        # Orchestration processing step.
        prompt = self.decision_prompt.format(
            case_id=state.get("case_id") or "unknown",
            completed_tasks=", ".join(completed_tasks) if completed_tasks else "",
            hypothesis_summary=self.memory.get_hypothesis_summary(),
            completed_exams=", ".join(completed_exams) if completed_exams else "",
            confidence=max_confidence,
            chief_complaint=state.get("chief_complaint", ""),
            diagnosis_result=", ".join(state.get("diagnosis_result", [])) if state.get("diagnosis_result") else ""
        )

        response = call_llm_api("You are a medical orchestrator.", prompt)

        result = parse_json_from_response(response)

        if result:
            return result
        else:
            return {
                "sufficient_for_diagnosis": False,
                "missing_info": [],
                "next_task": "continue",
                "reasoning": "",
            }

    def _find_gt_match(self, exam: str, gt_keys: set) -> Optional[str]:
        """Find the matching ground-truth exam key."""
        if not gt_keys:
            return None

        exam_lower = exam.lower()

        if exam in gt_keys:
            return exam

        for gt_key in gt_keys:
            gt_key_lower = gt_key.lower()
            if gt_key_lower in exam_lower or exam_lower in gt_key_lower:
                return gt_key

        return None

    def _should_recheck_with_llm(self, diagnosis_result: Dict) -> bool:
        """Decide whether diagnosis should trigger recheck."""
        missing_evidence = diagnosis_result.get("missing_evidence", [])

        if not missing_evidence:
            return False

        state = self.memory.get_all()
        if int(state.get("recheck_total", 0) or 0) >= int(state.get("max_recheck", 0) or 0):
            return False

        top_conf = float(state.get("top_hypothesis_confidence", 0.0) or 0.0)
        has_diagnosis = bool(diagnosis_result.get("diagnosis_result"))
        if has_diagnosis and top_conf >= 0.85:
            return False

        # Orchestrator-level gate; not directly tied to agent `need_recheck`.
        decision = self._make_llm_decision()

        if isinstance(decision, dict):
            if decision.get("sufficient_for_diagnosis", False):
                print("[log omitted: encoding-fixed]")
                return False

            if decision.get("missing_info"):
                print("[orchestrator] status update")
        else:
            print(f"  - warning: _make_llm_decision returned non-dict: {type(decision)}")
            return has_diagnosis and top_conf < 0.85

        return True
