"""
Treatment Agent - 治疗Agent
负责Task5: 根据所有信息给出治疗方案
"""
import re
from typing import Dict, List
from utils import load_prompt, call_llm_api, parse_json_from_response


# 有效的治疗方案列表
VALID_TREATMENTS = [
    "手术", "介入治疗", "药物治疗", "化学治疗", "抗生素治疗",
    "放射治疗", "物理疗法", "免疫疗法", "心理治疗", "中医治疗", "基因治疗"
]


class TreatmentAgent:
    """治疗Agent"""

    def __init__(self):
        self.prompt_template = load_prompt("treatment")

    def run(self, personal_info: Dict, chief_complaint: str, present_illness: str,
            past_history: str, examination_results: str, department: str,
            diagnosis_result: List[str], case_id: str = "未知") -> Dict:
        """
        执行治疗方案制定任务

        Args:
            personal_info: 个人信息
            chief_complaint: 主诉
            present_illness: 现病史
            past_history: 既往史
            examination_results: 查体结果
            department: 就诊科室
            diagnosis_result: 诊断结果列表

        Returns:
            {
                "treatment_plan": [str],  # 治疗方案列表
                "reasoning": str,  # 理由
                "raw_output": str
            }
        """
        # 构建prompt
        prompt = self.prompt_template.format(
            case_id=case_id,
            department=department,
            personal_info=str(personal_info),
            chief_complaint=chief_complaint,
            present_illness=present_illness,
            past_history=past_history if past_history else "无",
            examination_results=examination_results if examination_results else "无",
            diagnosis_result=", ".join(diagnosis_result) if diagnosis_result else "无"
        )

        # 调用LLM
        response = call_llm_api(f"你是一名{department}医生。", prompt)

        # 解析结果
        result = parse_json_from_response(response)

        if result and isinstance(result, dict):
            reasoning = result.get("reasoning", "")

            raw_treatments = result.get("treatment_plan")
            treatment_plan = self._coerce_treatments(
                raw_treatments,
                fallback_text=response,
                reasoning=reasoning
            )

            return {
                "treatment_plan": treatment_plan,
                "reasoning": reasoning,
                "raw_output": response
            }

        # 如果解析失败或返回的是列表/字符串，尝试从原始响应中提取治疗方案
        treatment_plan = self._coerce_treatments(
            result,
            fallback_text=response,
            reasoning=""
        )
        return {
            "treatment_plan": treatment_plan,
            "reasoning": "",
            "raw_output": response
        }

    def _validate_treatments(self, treatments: List[str]) -> List[str]:
        """验证治疗方案是否在有效列表中"""
        valid = []
        for t in treatments:
            t_clean = t.strip()
            # 精确匹配
            if t_clean in VALID_TREATMENTS:
                valid.append(t_clean)
            else:
                # 模糊匹配
                for valid_t in VALID_TREATMENTS:
                    if valid_t in t_clean or t_clean in valid_t:
                        if valid_t not in valid:
                            valid.append(valid_t)
                        break
        return valid

    def _extract_treatments_from_text(self, text: str) -> List[str]:
        """从文本中提取治疗方案"""
        if not text:
            return []

        treatments = []
        # 使用逗号、顿号、换行符分割
        parts = re.split(r'[，,、\n]', text)
        parts = [p.strip() for p in parts if p.strip()]

        for part in parts:
            for valid_t in VALID_TREATMENTS:
                if valid_t in part:
                    if valid_t not in treatments:
                        treatments.append(valid_t)

        return treatments

    def _coerce_treatments(
        self,
        raw_treatments,
        fallback_text: str,
        reasoning: str
    ) -> List[str]:
        """将模型输出规范化为有效治疗方案列表"""
        if isinstance(raw_treatments, list):
            treatments = self._validate_treatments(raw_treatments)
        elif isinstance(raw_treatments, str):
            treatments = self._extract_treatments_from_text(raw_treatments)
            if treatments:
                treatments = self._validate_treatments(treatments)
        else:
            treatments = []

        if not treatments:
            treatments = self._extract_treatments_from_text(fallback_text)
            if treatments:
                treatments = self._validate_treatments(treatments)

        if not treatments and reasoning:
            treatments = self._extract_treatments_from_text(reasoning)
            if treatments:
                treatments = self._validate_treatments(treatments)

        return treatments
