"""Medical Diagnosis Multi-Agent Benchmark entry point."""
import json
import os
import random
import re
from datetime import datetime
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

from app_config import (
    BATCH_SIZE,
    BENCHMARK_MAX_SAMPLES,
    BENCHMARK_SAMPLE_POSITION,
    BENCHMARK_SEED,
    CASES_DIR,
    DEBUG,
    DEFAULT_DATASET_PATH,
    DOCLENS_AGGREGATION,
    IMG_BASE_DIR,
    MAX_RECHECK_PER_CASE,
    MCTS_PLANNING_CONFIG,
    MCTS_ROLLOUT_CALL_CAP_PER_SEARCH,
    MCTS_ROLLOUT_CALL_CAP_TOTAL,
    MCTS_ROLLOUT_CONFIG,
    PROJECT_ROOT,
)
from utils import (
    extract_ground_truth, calculate_accuracy, calculate_iou,
    safe_mean, judge_api_messages, load_prompt, extract_statements_from_response,
    count_entailments, extract_entailment_predictions, normalize_imaging_type,
    parse_json_from_response,
    start_case_llm_tracking, stop_case_llm_tracking
)
from med_orchestrator import MedOrchestrator


# ==========================================
# Benchmark workflow step.
# ==========================================


# ==========================================
# Benchmark workflow step.
# ==========================================
def evaluate_task1_referral(state: dict, ground_truth: dict) -> dict:
    """Evaluate Task 1 referral results."""
    pred_l1 = state.get("dept_l1", "")
    pred_l2 = state.get("dept_l2", [])
    gt_l1 = ground_truth.get("gt_dept_l1", "")
    gt_l2 = ground_truth.get("gt_dept_l2", [])

    acc_l1 = None if not str(gt_l1).strip() else calculate_accuracy(pred_l1, gt_l1)
    iou_l2 = calculate_iou(pred_l2, gt_l2)

    return {
        "l1_accuracy": acc_l1,
        "l2_iou": iou_l2,
        "pred_l1": pred_l1,
        "pred_l2": pred_l2,
        "gt_l1": gt_l1,
        "gt_l2": gt_l2
    }


def evaluate_task2_doctor(state: dict, ground_truth: dict) -> dict:
    """Evaluate Task 2 exam recommendation results."""
    # Benchmark workflow step.
    pred_physical = state.get("predicted_physical_exams", [])
    pred_auxiliary = state.get("predicted_auxiliary_exams", [])
    pred_exams = pred_physical + pred_auxiliary

    gt_exams = ground_truth.get("gt_exams", [])

    iou = calculate_iou(pred_exams, gt_exams)

    return {
        "exam_iou": iou,
        "pred_exams": pred_exams,
        "gt_exams": gt_exams
    }


def evaluate_task3_imaging(state: dict, ground_truth: dict) -> dict:
    """Evaluate Task 3 imaging report results."""
    imaging_results = state.get("imaging_results", {})
    gt_img_reports = ground_truth.get("gt_img_reports", {})

    if not gt_img_reports:
        return {"score": None, "skipped": True}
    if not imaging_results:
        # These benchmark cases contain a reference image report, so failing to
        # order/interpret it is an applicable Task-3 miss, not a missing metric.
        return {
            "score": 0.0,
            "skipped": False,
            "scores": {},
            "pred_reports": {},
            "gt_reports": gt_img_reports,
        }

    # Some MedChain records provide one combined reference report rather than
    # modality-keyed reports. Evaluate the combined prediction once in that
    # case; otherwise every valid CT/MRI prediction would appear unmatched.
    if set(gt_img_reports) == {"ALL"}:
        combined_prediction = "\n".join(
            str(report) for report in imaging_results.values() if report
        )
        if not combined_prediction:
            return {
                "score": 0.0,
                "skipped": False,
                "scores": {},
                "pred_reports": {},
                "gt_reports": gt_img_reports,
            }
        combined_score = evaluate_radiology_doclens(
            combined_prediction,
            gt_img_reports["ALL"],
        )
        return {
            "score": combined_score.get("score", 0.0),
            "skipped": False,
            "scores": {"ALL": combined_score},
            "pred_reports": {"ALL": combined_prediction},
            "gt_reports": gt_img_reports,
        }

    per_type_scores = {}
    pred_reports = {}
    matched_gt_reports = {}
    normalized_predictions = {
        normalize_imaging_type(exam_type): pred_report
        for exam_type, pred_report in imaging_results.items()
        if pred_report
    }

    # Score every applicable reference modality. A missing generated modality
    # contributes zero instead of disappearing from the average.
    for gt_type, gt_report in gt_img_reports.items():
        normalized_type = normalize_imaging_type(gt_type)
        pred_report = normalized_predictions.get(normalized_type)
        matched_gt_reports[normalized_type] = gt_report
        if not pred_report:
            per_type_scores[normalized_type] = {
                "score": 0.0,
                "claim_recall": 0.0,
                "claim_precision": 0.0,
                "gt_claims": [],
                "pred_claims": [],
            }
            continue
        score = evaluate_radiology_doclens(pred_report, gt_report)
        per_type_scores[normalized_type] = score
        pred_reports[normalized_type] = pred_report

    def average_metric(metric_name: str):
        values = [
            v.get(metric_name) for v in per_type_scores.values()
            if v.get(metric_name) is not None
        ]
        return safe_mean(values) if values else None

    score = average_metric("score")

    return {
        "score": score,
        "scores": per_type_scores,
        "pred_reports": pred_reports,
        "gt_reports": matched_gt_reports
    }

def _build_doclens_claim_prompt(report: str):
    prompt = load_prompt("doclens_extract")
    prompt = prompt.format(report=report)
    return [{"role": "user", "content": prompt}]


def _build_doclens_entail_prompt(report: str, claims: list):
    prompt = load_prompt("doclens_entail")
    claims_text = "\n".join(claims) if isinstance(claims, list) else claims

    # Benchmark workflow step.
    prompt = prompt.replace("{gt_claims}", claims_text).replace("{pred_report}", report)
    return [{"role": "user", "content": prompt}]


def _parse_doclens_claims(response_text: str):
    data = parse_json_from_response(response_text)
    if isinstance(data, dict):
        data = data.get("claims")
    if isinstance(data, list):
        claims = []
        for item in data:
            if isinstance(item, str):
                value = item.strip()
            elif isinstance(item, dict):
                value = str(item.get("claim", "")).strip()
            else:
                value = ""
            if value:
                claims.append(value)
        if claims:
            return claims
    return extract_statements_from_response(response_text)


def _split_report_sentences(report: str):
    if not report:
        return []
    sentences = re.split(r'(?<=[。.!?])\s+|\n+', report)
    return [s.strip() for s in sentences if s.strip()]


def _doclens_entail_claims(clinical_report: str, claims: list):
    if not claims:
        return 0.0
    entail_prompt = _build_doclens_entail_prompt(clinical_report, claims)
    recall_resp = judge_api_messages(entail_prompt)
    entailment_predictions = extract_entailment_predictions(recall_resp)
    if entailment_predictions:
        return sum(entailment_predictions) / len(claims)
    recall_entailed = count_entailments(recall_resp, len(claims))
    return recall_entailed / len(claims)
    # return 1


def evaluate_radiology_doclens(pred_report: str, gt_report: str) -> dict:
    """Evaluate DocLens claim recall/precision and their harmonic mean."""
    if not gt_report or not pred_report:
        return {
            "score": 0.0,
            "claim_recall": 0.0,
            "claim_precision": 0.0,
            "gt_claims": [],
            "pred_claims": [],
        }

    gt_extract_resp = judge_api_messages(_build_doclens_claim_prompt(gt_report))
    gt_claims = _parse_doclens_claims(gt_extract_resp)
    if not gt_claims:
        return {
            "score": 0.0,
            "claim_recall": 0.0,
            "claim_precision": 0.0,
            "gt_claims": [],
            "pred_claims": [],
        }

    pred_extract_resp = judge_api_messages(_build_doclens_claim_prompt(pred_report))
    pred_claims = _parse_doclens_claims(pred_extract_resp)

    claim_recall = _doclens_entail_claims(pred_report, gt_claims)
    claim_precision = (
        _doclens_entail_claims(gt_report, pred_claims) if pred_claims else 0.0
    )
    denom = claim_recall + claim_precision
    claim_f1 = (
        2.0 * claim_recall * claim_precision / denom if denom > 0 else 0.0
    )
    aggregate_scores = {
        "recall": claim_recall,
        "precision": claim_precision,
        "mean": (claim_recall + claim_precision) / 2.0,
        "f1": claim_f1,
    }
    aggregate_name = (
        DOCLENS_AGGREGATION if DOCLENS_AGGREGATION in aggregate_scores else "f1"
    )

    return {
        "score": aggregate_scores[aggregate_name],
        "aggregation": aggregate_name,
        "claim_recall": claim_recall,
        "claim_precision": claim_precision,
        "gt_claims": gt_claims,
        "pred_claims": pred_claims,
    }


def evaluate_task4_diagnosis(state: dict, ground_truth: dict) -> dict:
    """Evaluate Task 4 diagnosis results."""
    pred_diag = state.get("diagnosis_result", [])
    gt_diag = ground_truth.get("gt_diag", "")

    if not gt_diag:
        return {"score": None, "raw_score": None, "pred": "", "gt": gt_diag}
    if not pred_diag:
        return {"score": 0.2, "raw_score": 1.0, "pred": "", "gt": gt_diag}

    pred_str = ", ".join(pred_diag) if isinstance(pred_diag, list) else str(pred_diag)

    # Benchmark workflow step.
    eval_prompt = load_prompt("diagnosis_eval")
    prompt = eval_prompt.format(gt=gt_diag, test=pred_str)

    response = judge_api_messages([
        {"role": "system", "content": "You are a medical evaluator."},
        {"role": "user", "content": [{"type": "text", "text": prompt}]}
    ])

    try:
        raw_score = float(max(1, min(5, int(re.search(r'\d+', response).group()))))
    except:
        raw_score = 1.0

    return {
        # Paper Sec. 4.1 normalizes the 1-5 judge score to [0, 1].
        "score": raw_score / 5.0,
        "raw_score": raw_score,
        "pred": pred_str,
        "gt": gt_diag
    }


def evaluate_task5_treatment(state: dict, ground_truth: dict) -> dict:
    """Evaluate Task 5 treatment results."""
    pred_treat = state.get("treatment_plan", [])
    gt_treat = ground_truth.get("gt_treat", [])

    iou = calculate_iou(pred_treat, gt_treat)

    return {
        "treatment_iou": iou,
        "pred": pred_treat,
        "gt": gt_treat
    }


def sanitize_filename(name: str) -> str:
    """Return a Windows-safe filename."""
    # Benchmark workflow step.
    invalid_chars = '<>:"/\\|?*'
    for char in invalid_chars:
        name = name.replace(char, '_')
    # Benchmark workflow step.
    if len(name) > 100:
        name = name[:100]
    return name


def _to_int(value, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def build_orchestration_usage_summary(logs: list, dataset_path: str) -> dict:
    """Build per-case orchestration call-count table and aggregate stats."""
    rows = []

    for log in logs:
        usage = log.get("llm_usage", {}) if isinstance(log, dict) else {}
        metrics = log.get("metrics", {}) if isinstance(log, dict) else {}
        has_error = isinstance(log, dict) and ("error" in log)

        row = {
            "case_id": log.get("id") if isinstance(log, dict) else None,
            "success": 0 if has_error else 1,
            "recheck_count": _to_int(log.get("recheck_total", metrics.get("recheck_count", 0))),
            "llm_calls_total": _to_int(usage.get("llm_calls_total", metrics.get("llm_calls", 0))),
            "llm_calls_base": _to_int(usage.get("llm_calls_base", 0)),
            "llm_calls_recheck": _to_int(usage.get("llm_calls_recheck", 0)),
            "rollout_llm_calls": _to_int(usage.get("rollout_calls_total", metrics.get("rollout_llm_calls", 0))),
            "rollout_llm_calls_base": _to_int(usage.get("rollout_calls_base", 0)),
            "rollout_llm_calls_recheck": _to_int(usage.get("rollout_calls_recheck", 0)),
            "llm_error_calls": _to_int(usage.get("errors", 0)),
            "api_calls_call_llm_api": _to_int(usage.get("api_calls", {}).get("call_llm_api", 0)),
            "api_calls_img_api": _to_int(usage.get("api_calls", {}).get("img_api", 0)),
            "api_calls_judge_api": _to_int(usage.get("api_calls", {}).get("judge_api", 0)),
            "api_calls_judge_api_messages": _to_int(usage.get("api_calls", {}).get("judge_api_messages", 0)),
            "api_calls_mcts_planning_llm": _to_int(usage.get("api_calls", {}).get("mcts_planning_llm", 0)),
            "api_calls_mcts_rollout_simulation_llm": _to_int(
                usage.get("api_calls", {}).get("mcts_rollout_simulation_llm", 0)
            ),
        }
        row["non_rollout_llm_calls"] = max(0, row["llm_calls_total"] - row["rollout_llm_calls"])
        rows.append(row)

    success_rows = [row for row in rows if row["success"] == 1]

    numeric_keys = [
        "recheck_count",
        "llm_calls_total",
        "llm_calls_base",
        "llm_calls_recheck",
        "rollout_llm_calls",
        "rollout_llm_calls_base",
        "rollout_llm_calls_recheck",
        "non_rollout_llm_calls",
        "llm_error_calls",
        "api_calls_call_llm_api",
        "api_calls_img_api",
        "api_calls_judge_api",
        "api_calls_judge_api_messages",
        "api_calls_mcts_planning_llm",
        "api_calls_mcts_rollout_simulation_llm",
    ]

    aggregate_sum = {key: sum(row.get(key, 0) for row in success_rows) for key in numeric_keys}
    aggregate_mean = {key: safe_mean([row.get(key, 0) for row in success_rows], default=0.0) for key in numeric_keys}

    return {
        "schema_version": "1.1",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "dataset_path": dataset_path,
        "n_cases_total": len(rows),
        "n_cases_success": len(success_rows),
        "columns": ["case_id", "success"] + numeric_keys,
        "rows": rows,
        "aggregate_sum_success_cases": aggregate_sum,
        "aggregate_mean_success_cases": aggregate_mean,
    }


def process_single_case(item, img_base_dir: str, held_out_case_ids=None) -> tuple:
    """Run one case through orchestration and evaluation."""
    sample_id, sample_content = item

    # Benchmark workflow step.
    ground_truth = extract_ground_truth(sample_content, img_base_dir)

    # Benchmark workflow step.
    os.makedirs(CASES_DIR, exist_ok=True)

    # Benchmark workflow step.
    safe_id = sanitize_filename(str(sample_id))
    memory_path = os.path.join(CASES_DIR, f"{safe_id}.json")
    orchestrator = MedOrchestrator(
        memory_path=memory_path,
        img_base_dir=img_base_dir,
        mcts_planning_config=MCTS_PLANNING_CONFIG,
        mcts_rollout_config=MCTS_ROLLOUT_CONFIG,
        mcts_rollout_call_cap_per_search=MCTS_ROLLOUT_CALL_CAP_PER_SEARCH,
        mcts_rollout_call_cap_total=MCTS_ROLLOUT_CALL_CAP_TOTAL,
        max_recheck=MAX_RECHECK_PER_CASE,
        held_out_case_ids=held_out_case_ids,
    )
    start_case_llm_tracking(case_id=str(sample_id))
    llm_usage = None

    try:
        # Benchmark workflow step.
        final_state = orchestrator.run(ground_truth, case_id=sample_id)
        llm_usage = stop_case_llm_tracking()
        recheck_total = final_state.get("recheck_total", 0)

        # Benchmark workflow step.
        eval_results = {
            "task1": evaluate_task1_referral(final_state, ground_truth),
            "task2": evaluate_task2_doctor(final_state, ground_truth),
            "task3": evaluate_task3_imaging(final_state, ground_truth),
            "task4": evaluate_task4_diagnosis(final_state, ground_truth),
            "task5": evaluate_task5_treatment(final_state, ground_truth)
        }

        # Benchmark workflow step.
        metrics = {
            "t1_acc": eval_results["task1"]["l1_accuracy"],
            "t1_iou": eval_results["task1"]["l2_iou"],
            "t2_iou": eval_results["task2"]["exam_iou"],
            "t3_score": eval_results["task3"].get("score"),
            "t4_score": eval_results["task4"]["score"],
            "t5_iou": eval_results["task5"]["treatment_iou"],
            "llm_calls": llm_usage.get("llm_calls_total", 0),
            "recheck_count": recheck_total,
            "rollout_llm_calls": llm_usage.get("rollout_calls_total", 0),
        }
        paper_metric_values = [
            metrics.get("t1_acc"), metrics.get("t1_iou"), metrics.get("t2_iou"),
            metrics.get("t3_score"), metrics.get("t4_score"), metrics.get("t5_iou"),
        ]
        paper_metric_values = [value for value in paper_metric_values if value is not None]
        metrics["paper_average"] = safe_mean(paper_metric_values) if paper_metric_values else None

        # Benchmark workflow step.
        log = {
            "id": sample_id,
            "case_file": memory_path,
            "evaluations": eval_results,
            "metrics": metrics,
            "llm_usage": llm_usage,
            "recheck_total": recheck_total,
        }

        return metrics, log

    except Exception as e:
        if llm_usage is None:
            llm_usage = stop_case_llm_tracking()
        import traceback
        traceback.print_exc()
        return {}, {"id": sample_id, "error": str(e), "llm_usage": llm_usage}


def run_benchmark(
    dataset_path: str,
    max_samples: int = None,
    seed: int = 42,
    skip_count: int = 0,
    sample_count: int = None,
    sample_position: str = "head",
):
    """Run the configured benchmark."""
    with open(dataset_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    items = list(data.items())
    if skip_count:
        items = items[skip_count:]

    random.seed(seed)
    requested_count = sample_count if sample_count is not None else max_samples
    if requested_count is not None:
        requested_count = max(0, min(int(requested_count), len(items)))
        position = (sample_position or "head").lower()
        if position == "random":
            items = random.sample(items, requested_count)
        elif position == "tail":
            items = items[-requested_count:] if requested_count else []
        else:
            items = items[:requested_count]

    total_metrics = defaultdict(list)
    logs = []
    held_out_case_ids = {str(case_id) for case_id, _ in items}

    print(f"\nStarting benchmark with {len(items)} cases...")
    print(f"Dataset: {dataset_path}")
    print(f"DEBUG mode: {DEBUG}")

    with ThreadPoolExecutor(max_workers=BATCH_SIZE) as executor:
        futures = {
            executor.submit(
                process_single_case,
                item,
                IMG_BASE_DIR,
                held_out_case_ids,
            ): item[0]
            for item in items
        }

        for future in tqdm(as_completed(futures), total=len(items), desc="Processing"):
            case_id = futures[future]
            try:
                metrics, log = future.result()
                logs.append(log)
                for key, value in metrics.items():
                    if value is not None:
                        total_metrics[key].append(value)
            except Exception as e:
                print(f"[{case_id}] processing error: {e}")
                logs.append({"id": case_id, "error": str(e)})

    print("\n" + "=" * 60)
    print("BENCHMARK RESULTS")
    print("=" * 60)

    print("\nTask 1 (Referral):")
    print(f"  - L1 Accuracy: {safe_mean(total_metrics['t1_acc']):.4f} (n={len(total_metrics['t1_acc'])})")
    print(f"  - L2 IoU: {safe_mean(total_metrics['t1_iou']):.4f} (n={len(total_metrics['t1_iou'])})")

    print("\nTask 2 (Doctor/Exam Planning):")
    print(f"  - Exam IoU: {safe_mean(total_metrics['t2_iou']):.4f} (n={len(total_metrics['t2_iou'])})")

    if total_metrics["t3_score"]:
        print("\nTask 3 (Imaging):")
        print(f"  - DocLens Score: {safe_mean(total_metrics['t3_score']):.4f} (n={len(total_metrics['t3_score'])})")
    else:
        print("\nTask 3 (Imaging): N/A (no samples)")

    print("\nTask 4 (Diagnosis):")
    print(f"  - Judge Score: {safe_mean(total_metrics['t4_score']):.4f} (n={len(total_metrics['t4_score'])})")

    print("\nTask 5 (Treatment):")
    print(f"  - Treatment IoU: {safe_mean(total_metrics['t5_iou']):.4f} (n={len(total_metrics['t5_iou'])})")

    if total_metrics["paper_average"]:
        print("\nPaper-normalized average:")
        print(
            f"  - Average: {safe_mean(total_metrics['paper_average']):.4f} "
            f"(n={len(total_metrics['paper_average'])})"
        )

    print("\n" + "=" * 60)

    success_count = sum(1 for log in logs if "error" not in log)
    print(f"\nProcessing stats: total {len(logs)} | success {success_count} | failed {len(logs) - success_count}")

    results_path = os.path.join(PROJECT_ROOT, "benchmark_results.json")
    with open(results_path, "w", encoding='utf-8') as f:
        json.dump(logs, f, ensure_ascii=False, indent=2)
    print(f"Results saved to: {results_path}")

    usage_summary = build_orchestration_usage_summary(logs, dataset_path)
    usage_path = os.path.join(PROJECT_ROOT, "orchestration_usage_summary.json")
    with open(usage_path, "w", encoding='utf-8') as f:
        json.dump(usage_summary, f, ensure_ascii=False, indent=2)
    print(f"Orchestration usage summary saved to: {usage_path}")

    return total_metrics, logs


def main():
    """Run the benchmark with centralized configuration."""
    dataset_path = DEFAULT_DATASET_PATH
    run_benchmark(
        dataset_path=dataset_path,
        max_samples=BENCHMARK_MAX_SAMPLES,
        sample_position=BENCHMARK_SAMPLE_POSITION,
        seed=BENCHMARK_SEED,
    )


if __name__ == "__main__":
    main()
