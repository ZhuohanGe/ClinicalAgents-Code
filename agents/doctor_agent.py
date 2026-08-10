"""
Doctor Agent - 问诊Agent
负责Task2: 根据病人信息给出疾病假设和检查建议
"""
from typing import Dict, List, Optional
from utils import load_prompt, call_llm_api, parse_json_from_response
import re


# 有效的体格检查列表
VALID_PHYSICAL_EXAMS = [
    "一般检查", "头颅眼耳鼻喉检查", "颈部检查", "胸部检查", "腹部检查",
    "脊柱和四肢检查", "皮肤检查", "神经系统检查", "泌尿生殖系统检查"
]

# 有效的辅助检查列表
VALID_AUXILIARY_EXAMS = [
    "X-ray", "MRI", "CT", "超声", "核医学成像",
    "血液学检查", "尿液检查", "粪便检查", "内镜检查", "病理检查"
]


class DoctorAgent:
    """问诊Agent"""

    def __init__(self):
        self.prompt_template = load_prompt("doctor")

    def run(self, personal_info: Dict, chief_complaint: str, present_illness: str,
            dept_l1: str, past_history: str, case_id: str = "未知",
            completed_physical_exams: List[str] = None,
            completed_auxiliary_exams: List[str] = None) -> Dict:
        """
        执行问诊任务

        Args:
            personal_info: 个人信息
            chief_complaint: 主诉
            present_illness: 现病史
            dept_l1: 一级科室
            dept_l2: 二级科室列表
            past_history: 既往史
            completed_physical_exams: 已完成的体格检查（用于回溯时过滤）
            completed_auxiliary_exams: 已完成的辅助检查（用于回溯时过滤）

        Returns:
            {
                "hypothesis_illness": [
                    {"disease": str, "evidence": [str], "confidence": float}
                ],
                "physical_exams": [str],
                "auxiliary_exams": [str],
                "raw_output": str
            }
        """
        # 构建过滤说明
        filter_note = ""
        if completed_physical_exams or completed_auxiliary_exams:
            filter_parts = []
            if completed_physical_exams:
                filter_parts.append(f"已完成的体格检查（无需重复）: {', '.join(completed_physical_exams)}")
            if completed_auxiliary_exams:
                filter_parts.append(f"已完成的辅助检查（无需重复）: {', '.join(completed_auxiliary_exams)}")
            filter_note = "注意：\n" + "\n".join(filter_parts)

        # 构建prompt
        prompt = self.prompt_template.format(
            case_id=case_id,
            department=dept_l1,
            personal_info=str(personal_info),
            chief_complaint=chief_complaint,
            present_illness=present_illness,
            # dept_l1=dept_l1,
            # dept_l2=", ".join(dept_l2) if dept_l2 else "无",
            past_history=past_history if past_history else "无",
            filter_note=filter_note
        )

        # 调用LLM
        response = call_llm_api(f"你是一名{dept_l1}医生。", prompt)

        # 解析结果
        result = parse_json_from_response(response)

        # 确保result是字典类型
        if result and isinstance(result, dict):
            hypothesis = result.get("hypothesis_illness", [])
            physical_exams = (
                result.get("physical_exams")
                or result.get("suggested_physical_exams")
                or result.get("physical_exam")
                or []
            )
            auxiliary_exams = (
                result.get("auxiliary_exams")
                or result.get("suggested_auxiliary_exams")
                or result.get("auxiliary_exam")
                or []
            )
        elif result and isinstance(result, list):
            # 如果返回的是列表，尝试将其作为hypothesis_illness处理
            print(f"    [DoctorAgent] 警告：LLM返回列表而非字典，尝试解析")
            hypothesis = result
            physical_exams = []
            auxiliary_exams = []
        else:
            # 解析失败时返回空结果
            return {
                "hypothesis_illness": [],
                "physical_exams": [],
                "auxiliary_exams": [],
                "raw_output": response
            }

        if isinstance(physical_exams, str):
            physical_exams = [physical_exams]
        if isinstance(auxiliary_exams, str):
            auxiliary_exams = [auxiliary_exams]

        # 去除医学上相近/类似的疾病，并限制最多8个假设
        hypothesis = self._dedupe_similar_hypotheses(hypothesis)
        if len(hypothesis) > 8:
            hypothesis = hypothesis[:8]

        # 验证检查项目是否在有效列表中
        physical_exams = self._validate_exams(physical_exams, VALID_PHYSICAL_EXAMS)
        auxiliary_exams = self._validate_exams(auxiliary_exams, VALID_AUXILIARY_EXAMS)

        # 过滤已完成的检查
        if completed_physical_exams:
            physical_exams = [e for e in physical_exams if e not in completed_physical_exams]
        if completed_auxiliary_exams:
            auxiliary_exams = [e for e in auxiliary_exams if e not in completed_auxiliary_exams]

        return {
            "hypothesis_illness": hypothesis,
            "physical_exams": physical_exams,
            "auxiliary_exams": auxiliary_exams,
            "raw_output": response
        }

    def _validate_exams(self, predicted_exams: List[str], valid_exams: List[str]) -> List[str]:
        """
        验证预测的检查项目是否在有效列表中，并规范化为标准名称

        规则：
        1. 精确匹配：直接在有效列表中
        2. 包含匹配：预测项包含有效项，或有效项包含预测项 → 返回标准有效项名称
        3. 否则抛弃

        Args:
            predicted_exams: LLM预测的检查项目列表
            valid_exams: 有效的检查项目列表

        Returns:
            验证后的检查项目列表（始终返回标准有效项名称，不返回LLM的冗长名称）
        """
        validated = []

        for pred in predicted_exams:
            if not pred or not isinstance(pred, str):
                continue

            pred_clean = pred.strip()
            pred_lower = pred_clean.lower()

            # 1. 精确匹配
            if pred_clean in valid_exams:
                if pred_clean not in validated:
                    validated.append(pred_clean)
                    print(f"    [DoctorAgent] 精确匹配: {pred_clean}")
                continue

            # 2. 包含匹配 - 始终返回标准有效项名称
            matched_valid = None
            for valid in valid_exams:
                valid_lower = valid.lower()
                # 预测项包含有效项，或有效项包含预测项
                if valid_lower in pred_lower or pred_lower in valid_lower:
                    matched_valid = valid
                    break

            if matched_valid:
                if matched_valid not in validated:
                    validated.append(matched_valid)
                    print(f"    [DoctorAgent] 规范化: '{pred_clean[:30]}...' -> '{matched_valid}'")
            else:
                # 3. 未匹配则抛弃
                print(f"    [DoctorAgent] 抛弃无效检查项: {pred_clean[:50]}...")

        return validated

    def _dedupe_similar_hypotheses(self, hypotheses: List[Dict]) -> List[Dict]:
        """
        去除医学上相近或重复的疾病（基于名称相似/包含）。
        """
        filtered = []
        seen = []
        for item in hypotheses:
            disease = item.get("disease", "")
            norm = self._normalize_disease_name(disease)
            if not norm:
                continue
            is_similar = False
            for existing in seen:
                if norm == existing or norm in existing or existing in norm:
                    is_similar = True
                    break
            if is_similar:
                continue
            filtered.append(item)
            seen.append(norm)
        return filtered

    def _normalize_disease_name(self, name: str) -> str:
        if not name or not isinstance(name, str):
            return ""
        cleaned = re.sub(r"[（(].*?[)）]", "", name)
        cleaned = re.sub(r"[\s,，;；、/\\-]", "", cleaned)
        return cleaned.lower()

    def get_physical_exam_from_gt(self, gt_physical_exam: Dict, required_exams: List[str]) -> Dict:
        """
        从ground truth中获取对应的体格检查结果

        Args:
            gt_physical_exam: ground truth中的体格检查结果
            required_exams: 需要的检查项目列表

        Returns:
            匹配到的体格检查结果字典
        """
        results = {}

        # 标准化检查名称进行匹配
        def normalize(s):
            return s.strip().lower().replace('-', '').replace('_', '').replace(' ', '')

        gt_normalized = {normalize(k): (k, v) for k, v in gt_physical_exam.items()}

        for exam in required_exams:
            exam_norm = normalize(exam)
            # 尝试精确匹配
            if exam_norm in gt_normalized:
                orig_key, value = gt_normalized[exam_norm]
                results[orig_key] = value
            else:
                # 尝试模糊匹配
                for gt_norm, (orig_key, value) in gt_normalized.items():
                    if exam_norm in gt_norm or gt_norm in exam_norm:
                        results[orig_key] = value
                        break

        return results
