"""Shared utility functions."""
import json
import re
import os
import base64
import mimetypes
import threading
from contextlib import contextmanager
from openai import OpenAI
from app_config import (
    API_TIMEOUT,
    JUDGE_CONFIG,
    JUDGE_MAX_TOKENS,
    LLM_CONFIG,
    LLM_MAX_TOKENS,
    LLM_TEMPERATURE,
    PROJECT_ROOT,
    PROMPTS_DIR,
)

_LLM_TRACK_LOCAL = threading.local()


def _new_case_llm_tracker(case_id=None):
    return {
        "case_id": case_id,
        "llm_calls_total": 0,
        "llm_calls_base": 0,
        "llm_calls_recheck": 0,
        "rollout_calls_total": 0,
        "rollout_calls_base": 0,
        "rollout_calls_recheck": 0,
        "api_calls": {
            "call_llm_api": 0,
            "img_api": 0,
            "judge_api": 0,
            "judge_api_messages": 0,
            "mcts_planning_llm": 0,
            "mcts_rollout_simulation_llm": 0,
        },
        "errors": 0,
    }


def start_case_llm_tracking(case_id=None):
    _LLM_TRACK_LOCAL.tracker = _new_case_llm_tracker(case_id=case_id)
    _LLM_TRACK_LOCAL.phase = "base"


def stop_case_llm_tracking():
    tracker = getattr(_LLM_TRACK_LOCAL, "tracker", None)
    if tracker is None:
        return _new_case_llm_tracker(case_id=None)
    result = dict(tracker)
    result["api_calls"] = dict(tracker.get("api_calls", {}))
    _LLM_TRACK_LOCAL.tracker = None
    _LLM_TRACK_LOCAL.phase = "base"
    return result


def get_case_llm_phase():
    return getattr(_LLM_TRACK_LOCAL, "phase", "base")


def set_case_llm_phase(phase: str):
    _LLM_TRACK_LOCAL.phase = "recheck" if phase == "recheck" else "base"


@contextmanager
def llm_phase(phase: str):
    previous = get_case_llm_phase()
    set_case_llm_phase(phase)
    try:
        yield
    finally:
        set_case_llm_phase(previous)


def _record_llm_usage(api_name: str, is_error: bool = False):
    tracker = getattr(_LLM_TRACK_LOCAL, "tracker", None)
    if tracker is None:
        return

    phase = "recheck" if get_case_llm_phase() == "recheck" else "base"
    tracker["llm_calls_total"] += 1
    if phase == "recheck":
        tracker["llm_calls_recheck"] += 1
    else:
        tracker["llm_calls_base"] += 1

    if api_name == "mcts_rollout_simulation_llm":
        tracker["rollout_calls_total"] += 1
        key = "rollout_calls_recheck" if phase == "recheck" else "rollout_calls_base"
        tracker[key] += 1

    tracker.setdefault("api_calls", {}).setdefault(api_name, 0)
    tracker["api_calls"][api_name] += 1
    if is_error:
        tracker["errors"] += 1


# ==========================================
# Utility processing step.
# ==========================================
def load_prompt(prompt_name):
    """Load a prompt template from the prompt directory."""
    prompt_path = os.path.join(PROMPTS_DIR, f"{prompt_name}_prompt.txt")
    if not os.path.exists(prompt_path):
        raise FileNotFoundError(f"Prompt file not found: {prompt_path}")
    with open(prompt_path, 'r', encoding='utf-8') as f:
        return f.read()


# ==========================================
# Utility processing step.
# ==========================================
def clean_reasoning_content(text):
    """Remove hidden reasoning text from model output."""
    if not text:
        return ""
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    text = re.sub(r'(?i)^thinking\s+process:.*?(?=\n\n)', '', text, flags=re.DOTALL)
    text = re.sub(r'^\s*\*Thinking\.\.\.\*(?:\n\s*>.*)+', '', text, flags=re.MULTILINE)
    text = re.sub(r'```(?:thinking|reasoning).*?```', '', text, flags=re.DOTALL)
    text = re.sub(r'(?i)^thinking\s+process:\s*', '', text)
    return text.strip()


# ==========================================
# Utility processing step.
# ==========================================
def call_llm_api(system_role, user_input, image_paths=None):
    """Call the default LLM API."""
    client = OpenAI(
        api_key=LLM_CONFIG["api_key"],
        base_url=LLM_CONFIG["base_url"]
    )

    try:
        messages = [{"role": "system", "content": system_role}]
        user_content = [{"type": "text", "text": user_input}]
        messages.append({"role": "user", "content": user_content})

        response = client.chat.completions.create(
            model=LLM_CONFIG["model"],
            messages=messages,
            temperature=LLM_TEMPERATURE,
            max_tokens=LLM_MAX_TOKENS,
            timeout=API_TIMEOUT,
        )
        _record_llm_usage("call_llm_api")
        return clean_reasoning_content(response.choices[0].message.content)
    except Exception as e:
        _record_llm_usage("call_llm_api", is_error=True)
        print(f"API Error: {e}")
        raise RuntimeError("Default LLM API call failed") from e


def call_llm_api_with_config(system_role, user_input, api_config, api_tag="call_llm_api_with_config"):
    """Call an OpenAI-compatible chat API using an explicit config."""
    if not api_config:
        api_config = LLM_CONFIG

    client = OpenAI(
        api_key=api_config["api_key"],
        base_url=api_config["base_url"]
    )

    try:
        messages = [{"role": "system", "content": system_role}]
        user_content = [{"type": "text", "text": user_input}]
        messages.append({"role": "user", "content": user_content})

        response = client.chat.completions.create(
            model=api_config["model"],
            messages=messages,
            temperature=LLM_TEMPERATURE,
            max_tokens=LLM_MAX_TOKENS,
            timeout=API_TIMEOUT,
        )
        _record_llm_usage(api_tag)
        return clean_reasoning_content(response.choices[0].message.content)
    except Exception as e:
        _record_llm_usage(api_tag, is_error=True)
        print(f"API Error: {e}")
        raise RuntimeError(f"LLM API call failed ({api_tag})") from e


def img_api(img_paths, user_input):
    """Call the multimodal API for image inputs."""
    client = OpenAI(
        api_key=LLM_CONFIG["api_key"],
        base_url=LLM_CONFIG["base_url"]
    )

    encoded_images = []
    for image_path in img_paths:
        mime_type, _ = mimetypes.guess_type(image_path)
        if not mime_type:
            raise ValueError(f"Could not determine the MIME type of the image: {image_path}")
        with open(image_path, "rb") as image_file:
            encoded_image = base64.b64encode(image_file.read()).decode('utf-8')
            encoded_images.append(f"data:{mime_type};base64,{encoded_image}")

    messages_content = [{'type': 'text', 'text': f'{user_input}'}]
    for url in encoded_images:
        messages_content.append({
            'type': 'image_url',
            'image_url': {'url': f'{url}'}
        })

    response = client.chat.completions.create(
        model=LLM_CONFIG["model"],
        temperature=LLM_TEMPERATURE,
        messages=[{'role': 'user', 'content': messages_content}],
        timeout=API_TIMEOUT,
    )
    _record_llm_usage("img_api")
    return clean_reasoning_content(str(response.choices[0].message.content))


def judge_api(system_role, user_input):
    """Call the judge LLM API."""
    client = OpenAI(
        api_key=JUDGE_CONFIG["api_key"],
        base_url=JUDGE_CONFIG["base_url"]
    )

    try:
        messages = [{"role": "system", "content": system_role}]
        user_content = [{"type": "text", "text": user_input}]
        messages.append({"role": "user", "content": user_content})

        response = client.chat.completions.create(
            model=JUDGE_CONFIG["model"],
            messages=messages,
            temperature=LLM_TEMPERATURE,
            max_tokens=JUDGE_MAX_TOKENS,
            timeout=API_TIMEOUT,
        )
        _record_llm_usage("judge_api")
        return clean_reasoning_content(response.choices[0].message.content)
    except Exception as e:
        _record_llm_usage("judge_api", is_error=True)
        print(f"Judge API Error: {e}")
        raise RuntimeError("Judge API call failed") from e

def judge_api_messages(messages, max_tokens=JUDGE_MAX_TOKENS):
    """Call the judge LLM API with custom messages."""
    client = OpenAI(
        api_key=JUDGE_CONFIG["api_key"],
        base_url=JUDGE_CONFIG["base_url"]
    )

    try:
        response = client.chat.completions.create(
            model=JUDGE_CONFIG["model"],
            messages=messages,
            temperature=LLM_TEMPERATURE,
            max_tokens=max_tokens,
            timeout=API_TIMEOUT,
        )
        _record_llm_usage("judge_api_messages")
        return clean_reasoning_content(response.choices[0].message.content)
    except Exception as e:
        _record_llm_usage("judge_api_messages", is_error=True)
        print(f"Judge API Error: {e}")
        raise RuntimeError("Judge API message call failed") from e


# ==========================================
# Utility processing step.
# ==========================================
def parse_json_from_response(response_text):
    """Extract JSON from an LLM response."""
    if not response_text:
        return None

    # Utility processing step.
    text = re.sub(r'```json\s*', '', response_text)
    text = re.sub(r'```\s*', '', text)
    text = text.strip()

    # Utility processing step.
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Utility processing step.
    json_match = re.search(r'\{[\s\S]*\}', text)
    if json_match:
        try:
            return json.loads(json_match.group())
        except json.JSONDecodeError:
            pass

    # Utility processing step.
    json_match = re.search(r'\[[\s\S]*\]', text)
    if json_match:
        try:
            return json.loads(json_match.group())
        except json.JSONDecodeError:
            pass

    return None


def extract_statements_from_response(response_text):
    """Extract statement strings from an LLM response."""
    if not response_text:
        return []

    statements = []

    # Utility processing step.
    text = re.sub(r'```[a-z]*\n?', '', response_text)
    text = text.strip()

    # Utility processing step.
    json_match = re.search(r'\[[\s\S]*?\]', text)
    if json_match:
        try:
            parsed = json.loads(json_match.group())
            if isinstance(parsed, list) and all(isinstance(x, str) for x in parsed):
                statements = [s.strip() for s in parsed if s.strip()]
                if statements:
                    return statements
        except json.JSONDecodeError:
            pass

    # Utility processing step.
    numbered_pattern = r'^\s*\d+[\.\)]\s*(.+)$'
    lines = text.split('\n')
    for line in lines:
        match = re.match(numbered_pattern, line.strip())
        if match:
            stmt = match.group(1).strip()
            if stmt:
                statements.append(stmt)

    if statements:
        return statements

    # Utility processing step.
    claim_pattern = r'^\s*claim\s*\d+\s*[:\uff1a]\s*(.+)$'
    for line in lines:
        match = re.match(claim_pattern, line.strip(), flags=re.IGNORECASE)
        if match:
            stmt = match.group(1).strip()
            if stmt:
                statements.append(stmt)

    if statements:
        return statements

    # Utility processing step.
    bullet_pattern = r'^\s*[-\*\u2022]\s*(.+)$'
    for line in lines:
        match = re.match(bullet_pattern, line.strip())
        if match:
            stmt = match.group(1).strip()
            if stmt:
                statements.append(stmt)

    if statements:
        return statements

    # Utility processing step.
    sentences = re.split(r'[\u3002\.\n]+', text)
    statements = [s.strip() for s in sentences if len(s.strip()) > 10]

    return statements[:20]


# ==========================================
# Utility processing step.
# ==========================================
def calculate_accuracy(pred, gt):
    """Return exact-match accuracy."""
    return 1.0 if str(pred).strip() == str(gt).strip() else 0.0


def calculate_iou(pred_list, gt):
    """Calculate intersection-over-union for two lists."""
    if pred_list is None or pred_list == []:
        return 0.0
    if gt is None or gt == []:
        return None

    set1 = set(pred_list)
    set2 = set(gt)

    intersection = set1.intersection(set2)
    union = set1.union(set2)

    iou = len(intersection) / len(union) if union else 0.0
    return iou


def safe_mean(values, default=0.0):
    """Return the mean value, or a default for empty input."""
    if not values:
        return default
    return sum(values) / len(values)


def count_entailments(response, total):
    """Count entailment labels in a judge response."""
    entail_count = len(re.findall(r'\bENTAIL\b', response.upper()))
    similarly_count = len(re.findall(r'\bSIMILARLY\b', response.upper()))
    count = entail_count + similarly_count
    return min(count, total)

def extract_entailment_predictions(response_text):
    """Extract DocLens-style entailment predictions."""
    if not response_text:
        return []

    data = parse_json_from_response(response_text)
    predictions = []
    if isinstance(data, list):
        for item in data:
            if isinstance(item, dict):
                pred = item.get("entailment_prediction")
                if isinstance(pred, str):
                    pred = pred.strip()
                    try:
                        pred = float(pred)
                    except ValueError:
                        pred = None
                if isinstance(pred, (int, float)):
                    pred = max(0.0, min(1.0, float(pred)))
                    predictions.append(pred)
    return predictions


def normalize_imaging_type(name: str) -> str:
    """Normalize common imaging type names."""
    if not name:
        return name
    name_lower = str(name).lower()
    match_rules = {
        'X-ray': ['x\u7ebf', 'xray', 'x\u5149', 'x-ray', 'dr', 'cr'],
        'CT': ['ct', '\u7535\u8111\u65ad\u5c42', 'computed tomography'],
        'MRI': ['mri', '\u6838\u78c1', '\u78c1\u5171\u632f'],
        '\u8d85\u58f0': ['\u8d85\u58f0', 'b\u8d85', '\u5f69\u8d85', 'ultrasound'],
        '\u75c5\u7406\u68c0\u67e5': ['\u75c5\u7406', 'pathology'],
        '\u5185\u955c\u68c0\u67e5': ['\u5185\u955c', '\u80c3\u955c', '\u80a0\u955c', 'endoscopy'],
        '\u6838\u533b\u5b66\u6210\u50cf': ['\u6838\u533b\u5b66', 'pet', 'spect', 'nuclear']
    }

    for standard_name, aliases in match_rules.items():
        if any(alias in name_lower for alias in aliases):
            return standard_name

    return str(name)


def normalize_imaging_reports(report_data):
    """Normalize imaging reports into a mapping."""
    if report_data is None:
        return {}
    if isinstance(report_data, dict):
        normalized = {}
        for key, value in report_data.items():
            if value:
                normalized_key = normalize_imaging_type(key)
                normalized[normalized_key] = str(value)
        return normalized
    if isinstance(report_data, list):
        report_text = " ".join(str(x) for x in report_data if x)
        return {"ALL": report_text} if report_text else {}
    if isinstance(report_data, str):
        return {"ALL": report_data}
    return {}


# ==========================================
# Utility processing step.
# ==========================================
def join_str(data):
    """Convert a value or list into a string."""
    if isinstance(data, list):
        return " ".join(str(x) for x in data)
    return str(data) if data else "\u65e0"


def extract_ground_truth(sample_value, img_base_dir='./datasets/MedImg/'):
    """Extract ground-truth fields from a raw dataset sample."""
    info = sample_value.get('\u3010\u75c5\u6848\u4ecb\u7ecd\u3011', {})
    tags = sample_value.get('tags', {})
    summary = sample_value.get('\u3010\u75c5\u4f8b\u6458\u8981\u3011', [])

    # Utility processing step.
    # Utility processing step.
    personal_info = {"\u6027\u522b": "\u672a\u77e5", "\u5e74\u9f84": "\u672a\u77e5"}
    for item in summary:
        if isinstance(item, str) and '\u3010\u57fa\u672c\u4fe1\u606f\u3011' in item:
            # Utility processing step.
            basic_info = item.replace('\u3010\u57fa\u672c\u4fe1\u606f\u3011', '').strip()
            # Utility processing step.
            if basic_info:
                # Utility processing step.
                if basic_info.startswith('\u5973'):
                    personal_info["\u6027\u522b"] = "\u5973"
                elif basic_info.startswith('\u7537'):
                    personal_info["\u6027\u522b"] = "\u7537"

                # Utility processing step.
                import re
                age_match = re.search(r'(\d+)\s*\u5c81', basic_info)
                if age_match:
                    personal_info["\u5e74\u9f84"] = age_match.group(1) + "\u5c81"
            print(f"  [DEBUG] extracted personal_info from case summary: {personal_info}")
            break

    # Utility processing step.
    if personal_info["\u6027\u522b"] == "\u672a\u77e5":
        personal_info["\u6027\u522b"] = info.get('\u6027\u522b', '\u672a\u77e5')
    if personal_info["\u5e74\u9f84"] == "\u672a\u77e5":
        personal_info["\u5e74\u9f84"] = info.get('\u5e74\u9f84', '\u672a\u77e5')

    # Utility processing step.
    zhusu = join_str(info.get('\u4e3b\u8bc9', []))
    jiwangshi = join_str(info.get('\u65e2\u5f80\u53f2', []))
    xianbingshi = join_str(info.get('\u73b0\u75c5\u53f2', []))
    raw_img_report = info.get('\u5f71\u50cf\u62a5\u544a', [])
    gt_img_reports = normalize_imaging_reports(raw_img_report)
    yingxiangbaogao = join_str(raw_img_report)

    gt_dept_l1 = tags.get('\u79d1\u5ba4', [None])[0] if tags.get('\u79d1\u5ba4') else None
    gt_dept_l2 = tags.get('\u79d1\u5ba4', [])[1:] if len(tags.get('\u79d1\u5ba4', [])) > 1 else []

    physical = info.get('\u67e5\u4f53', {}).get('\u4f53\u683c\u68c0\u67e5', {})
    auxiliary = info.get('\u67e5\u4f53', {}).get('\u8f85\u52a9\u68c0\u67e5', {})

    def get_keys(d):
        return list(d.keys()) if isinstance(d, dict) else []

    gt_exams = get_keys(physical) + get_keys(auxiliary)
    chati_str = f"Physical exam: {physical}, Auxiliary exam: {auxiliary}"

    # Utility processing step.
    gt_img_paths = []
    img_pattern = re.compile(r'.*\.(jpg|png|jpeg|bmp)$', re.IGNORECASE)
    for item in summary:
        if isinstance(item, str) and img_pattern.match(item):
            full_path = os.path.join(img_base_dir, item)
            if os.path.exists(full_path):
                gt_img_paths.append(full_path)

    # Utility processing step.
    images_info = sample_value.get('\u56fe\u50cf', [])
    if not images_info:
        images_info = sample_value.get('images', [])
    if not images_info:
        images_info = sample_value.get('\u5f71\u50cf', [])
    if not images_info:
        images_info = sample_value.get('\u533b\u5b66\u5f71\u50cf', [])
    if not images_info:
        # Utility processing step.
        images_info = info.get('\u56fe\u50cf', [])
    if not images_info:
        images_info = info.get('\u5f71\u50cf', [])

    # Utility processing step.
    print(f"  [DEBUG] extracted images_info count: {len(images_info)}")
    print(f"  [DEBUG] extracted gt_img_paths count: {len(gt_img_paths)}")
    if gt_img_paths:
        print(f"  [DEBUG] gt_img_paths example: {gt_img_paths[0] if gt_img_paths else 'N/A'}")

    gt_img_report = yingxiangbaogao

    # Utility processing step.
    gt_diag_list = tags.get('\u75c5\u79cd', [])
    gt_diag = join_str(gt_diag_list)
    print(f"  [DEBUG] extracted gt_diag: {gt_diag[:100] if gt_diag else 'N/A'}...")

    gt_treat = sample_value.get('\u3010\u6cbb\u7597\u9879\u76ee\u3011', [])

    return {
        "personal_info": personal_info,
        "zhusu": zhusu,
        "jiwangshi": jiwangshi,
        "xianbingshi": xianbingshi,
        "chati_str": chati_str,
        "physical_exam": physical,
        "auxiliary_exam": auxiliary,
        "gt_exams": gt_exams,
        "gt_dept_l1": gt_dept_l1,
        "gt_dept_l2": gt_dept_l2,
        "gt_img_paths": gt_img_paths,
        "gt_img_report": gt_img_report,
        "gt_img_reports": gt_img_reports,
        "gt_diag": gt_diag,
        "gt_treat": gt_treat,
        "images_info": images_info
    }
