EXPANSION_SYSTEM = """You are a clinical orchestration controller.
Given the current working memory, score each available action by expected next-step value.
Return only a JSON array in this format:
[{"action": "action_name", "score": 0.0}, ...]
Do not include any extra text."""

EXPANSION_USER = """Patient profile: {personal_info}
Chief complaint: {chief_complaint}
Present illness: {present_illness}

Current evidence E_t:
- Completed physical exams: {completed_physical_exams}
- Completed auxiliary exams: {completed_auxiliary_exams}
- Completed imaging exams: {completed_imaging_exams}
- Physical exam results: {physical_exams}
- Auxiliary exam results: {auxiliary_exams}
- Imaging results: {imaging_results}

Current hypotheses H_t: {hypothesis_illness}
Current department: {dept_l1}
Action trajectory tau: {mcts_trajectory}
Current MCTS step: {mcts_step}
Experience memory K_t: {experience_knowledge}
Potential missing evidence E_t^p: {potential_missing_evidence}

Available actions:
{action_descriptions}

Score every action."""

ROLLOUT_POLICY_SYSTEM = """You are a clinical workflow orchestrator.
Given the current diagnosis state, select the single most valuable next action.
Return only the action name string."""

ROLLOUT_POLICY_USER = """Chief complaint: {chief_complaint}
Evidence summary: {evidence_summary}
Current hypotheses H_t: {hypothesis_illness}
Experience memory K_t: {experience_knowledge}
Potential missing evidence E_t^p: {potential_missing_evidence}
Action trajectory tau: {mcts_trajectory}
Available actions: {action_list}
Best next action:"""

SIMULATION_SYSTEM = """You are an independent counterfactual rollout simulator for a clinical workflow.
Given a patient state and one proposed action, predict the most likely post-action state.
You are not the professional agent that will execute the final selected action.
Never call, impersonate, or claim to have run ReferralAgent, DoctorAgent, ImagingAgent,
DiagnosisAgent, TreatmentAgent, retrieval tools, patient records, or ground-truth data.
Generate a prediction using only the state included in the prompt.
Return only a JSON object:
{
  "new_evidence": {
    "physical_exams": {"exam name": "finding"},
    "auxiliary_exams": {"exam name": "result"},
    "imaging_results": {"exam name": "finding"}
  },
  "updated_hypotheses": [
    {"name": "diagnosis name", "confidence": 0.0}
  ],
  "state_updates": {
    "dept_l1": "predicted department",
    "dept_l2": ["predicted subdepartment"],
    "predicted_physical_exams": ["exam name"],
    "predicted_auxiliary_exams": ["exam name"],
    "pending_auxiliary_exams": ["exam name"],
    "diagnosis_result": ["predicted diagnosis"],
    "treatment_plan": ["predicted treatment step"],
    "experience_knowledge": ["predicted retrieved knowledge"],
    "potential_missing_evidence": ["predicted evidence gap"]
  },
  "reasoning": "one sentence"
}
Only include fields plausibly changed by the proposed action. Empty fields may be omitted.
All values are simulated predictions, not observed clinical facts."""

SIMULATION_USER = """Patient: {chief_complaint}, {present_illness}
Current department: {dept_l1} / {dept_l2}
Current evidence E_t: {evidence_summary}
Current hypotheses H_t: {hypothesis_illness}
Pending auxiliary exams: {pending_auxiliary_exams}
Task status: {task_status}
Action trajectory tau: {mcts_trajectory}
Experience memory K_t: {experience_knowledge}
Potential missing evidence E_t^p: {potential_missing_evidence}
Action to simulate: {action} ({action_description})
Predicted result:"""

MISSING_EVIDENCE_SYSTEM = """You are a clinical quality-control expert.
Based on the current evidence and hypotheses, identify critical evidence still missing
to confirm or exclude the main hypotheses.
Return only a JSON string array, for example ["missing A", "missing B"].
If no critical evidence is missing, return [] only."""

MISSING_EVIDENCE_USER = """Chief complaint: {chief_complaint}
Collected evidence:
- Physical exams: {physical_exams}
- Auxiliary exams: {auxiliary_exams}
- Imaging results: {imaging_results}
- Past history: {past_history}
Current hypotheses H_t: {hypothesis_illness}
Experience memory K_t: {experience_knowledge}
Potential missing evidence E_t^p: {potential_missing_evidence}
Critical missing evidence:"""

BACKTRACK_SYSTEM = """You are a clinical workflow orchestrator responsible for diagnosis backtracking.
Given the current state, missing evidence, action trajectory, and explicitly indexed snapshots,
choose the snapshot/stage to restore and the next action after rollback.
Return only a JSON object:
{"target_snapshot_id": 0, "target_workflow_stage": "task stage", "target_action": "action_name", "reasoning": "brief reason"}
Use only a snapshot_id shown in the prompt. The target action will be validated after restoration."""

BACKTRACK_USER = """Current hypotheses H_t: {hypothesis_illness}
Missing evidence E_t^m: {missing_evidence}
Action trajectory tau: {mcts_trajectory}
Current MCTS step: {mcts_step}
Eligible historical snapshots: {snapshot_metadata}
Return the backtracking decision."""

HYPOTHESIS_CONFIDENCE_SYSTEM = """You are a clinical reasoning scorer.
Given patient information and current evidence, assign each candidate diagnosis a confidence score in [0, 1].
Return only a JSON array:
[{"name": "diagnosis name", "confidence": 0.0}, ...]"""

HYPOTHESIS_CONFIDENCE_USER = """Patient profile: {personal_info}
Chief complaint: {chief_complaint}
Present illness: {present_illness}
Physical exams: {physical_exams}
Auxiliary exams: {auxiliary_exams}
Imaging results: {imaging_results}
Candidate diagnoses: {hypothesis_names}
Return confidence scores."""
