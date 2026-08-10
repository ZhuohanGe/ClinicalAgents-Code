"""
Diagnosis Agent - 诊断Agent
负责Task4: 根据所有信息更新疾病假设并作出诊断
"""
import re
from typing import Dict, List
from utils import load_prompt, call_llm_api, parse_json_from_response


class DiagnosisAgent:
    """诊断Agent"""

    def __init__(self):
        self.prompt_template = load_prompt("diagnosis")

    def run(self, personal_info: Dict, chief_complaint: str, present_illness: str,
            dept_l1: str, past_history: str,
            examination_results: str, hypothesis_illness: List[Dict],
            case_id: str = "未知") -> Dict:
        """
        执行诊断任务

        Args:
            personal_info: 个人信息
            chief_complaint: 主诉
            present_illness: 现病史
            dept_l1: 一级科室
            dept_l2: 二级科室列表
            past_history: 既往史
            examination_results: 查体结果汇总（体格检查+辅助检查）
            hypothesis_illness: 当前疾病假设列表

        Returns:
            {
                "updated_hypothesis": [
                    {
                        "disease": str,
                        "evidence": [str],
                        "confidence": float,
                        "status": "confirmed/need_more_evidence/excluded"
                    }
                ],
                "diagnosis_result": [str],  # 确诊的疾病
                "missing_evidence": [
                    {
                        "disease": str,
                        "missing": [str],
                        "suggested_exams": [str]
                    }
                ],
                "need_recheck": bool,  # 是否需要回溯
                "raw_output": str
            }
        """
        # 构建假设疾病的字符串表示
        hypothesis_str = self._format_hypothesis(hypothesis_illness)

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
            examination_results=examination_results if examination_results else "无",
            hypothesis_illness=hypothesis_str
        )

        # 调用LLM
        response = call_llm_api(f"你是一名{dept_l1}医生。", prompt)

        # 解析结果
        result = parse_json_from_response(response)

        if result:
            updated_hypothesis = result.get("updated_hypothesis", [])
            diagnosis_result = result.get("diagnosis_result", [])
            missing_evidence = result.get("missing_evidence", [])
            need_recheck = result.get("need_recheck", False)
            missing_evidence = self._normalize_missing_evidence(missing_evidence)

            # 根据置信度自动判断是否需要复查
            if not need_recheck:
                need_recheck = self._check_need_recheck(updated_hypothesis)

            return {
                "updated_hypothesis": updated_hypothesis,
                "diagnosis_result": diagnosis_result,
                "missing_evidence": missing_evidence,
                "need_recheck": need_recheck,
                "raw_output": response
            }
        else:
            return {
                "updated_hypothesis": hypothesis_illness,
                "diagnosis_result": [],
                "missing_evidence": [],
                "need_recheck": False,
                "raw_output": response
            }

    def _format_hypothesis(self, hypothesis: List[Dict]) -> str:
        """格式化疾病假设列表"""
        if not hypothesis:
            return "无当前假设"

        parts = []
        for i, h in enumerate(hypothesis, 1):
            disease = h.get("disease", "未知")
            evidence = h.get("evidence", [])
            confidence = h.get("confidence", 0)
            evidence_str = ", ".join(evidence) if evidence else "无"
            parts.append(f"{i}. 疾病: {disease}\n   证据: [{evidence_str}]\n   置信度: {confidence}")

        return "\n".join(parts)

    def _check_need_recheck(self, hypothesis: List[Dict]) -> bool:
        """
        检查是否需要复查

        规则：
        - 置信度 >= 0.7: 确诊，不需要复查
        - 0.3 <= 置信度 < 0.7: 需要补充证据，需要复查
        - 置信度 < 0.3: 排除

        如果有任何疾病的置信度在0.3-0.7之间，则需要复查
        """
        for h in hypothesis:
            confidence = h.get("confidence", 0)
            if 0.3 <= confidence < 0.7:
                return True
        return False

    def _normalize_missing_evidence(self, missing_evidence: List[Dict]) -> List[Dict]:
        """
        将缺失证据规范为简短的“缺失检查”描述，避免冗余叙述。
        """
        normalized = []
        for item in missing_evidence:
            disease = item.get("disease", "").strip()
            suggested = item.get("suggested_exams") or []
            missing = item.get("missing") or []

            short_suggested = [
                self._short_exam_name(exam) for exam in suggested if exam
            ]
            short_suggested = [exam for exam in short_suggested if exam][:3]

            normalized_missing = []
            if short_suggested:
                normalized_missing = [f"需{exam}排除" for exam in short_suggested]
            else:
                for entry in missing[:3]:
                    exam_name = self._extract_exam_name(entry)
                    if exam_name:
                        normalized_missing.append(f"需{exam_name}排除")
                    else:
                        normalized_missing.append(self._short_text(entry))

            normalized.append({
                "disease": disease,
                "missing": normalized_missing,
                "suggested_exams": short_suggested
            })

        return normalized

    def _short_exam_name(self, exam: str) -> str:
        exam_clean = re.sub(r"[（(].*?[)）]", "", exam)
        exam_clean = re.split(r"[，,;；、/]", exam_clean)[0].strip()
        return exam_clean[:24]

    def _extract_exam_name(self, text: str) -> str:
        if not text:
            return ""
        keywords = [
            "胸片", "X-ray", "MRI", "CT", "超声", "病理", "内镜", "PET-CT",
            "血常规", "血液检查", "尿液检查", "粪便检查", "心电图", "肿瘤标志物"
        ]
        for keyword in keywords:
            if keyword.lower() in text.lower():
                return keyword
        return ""

    def _short_text(self, text: str) -> str:
        text = re.sub(r"\s+", "", text)
        return text[:24] if text else ""


    def get_suggested_exams_from_missing(self, missing_evidence: List[Dict]) -> Dict:
        """
        从missing_evidence中提取建议的检查

        Returns:
            {
                "physical_exams": [str],
                "auxiliary_exams": [str]
            }
        """
        physical_exams = []
        auxiliary_exams = []

        # 体格检查类型
        physical_types = ['一般检查', '头颅眼耳鼻喉检查', '颈部检查', '胸部检查',
                          '腹部检查', '脊柱和四肢检查', '皮肤检查', '神经系统检查',
                          '泌尿生殖系统检查']

        # 辅助检查类型
        auxiliary_types = ['X-ray', 'MRI', 'CT', '超声', '核医学成像',
                           '血液学检查', '尿液检查', '粪便检查', '内镜检查', '病理检查']

        for item in missing_evidence:
            suggested = item.get("suggested_exams", [])
            for exam in suggested:
                exam_lower = exam.lower()

                # 尝试匹配体格检查（返回标准名称）
                matched_physical = None
                for p in physical_types:
                    if p.lower() in exam_lower or exam_lower in p.lower():
                        matched_physical = p
                        break

                if matched_physical:
                    if matched_physical not in physical_exams:
                        physical_exams.append(matched_physical)
                        print(f"    [DiagnosisAgent] 体格检查规范化: '{exam[:30]}...' -> '{matched_physical}'")
                else:
                    # 尝试匹配辅助检查（返回标准名称）
                    matched_auxiliary = None
                    for a in auxiliary_types:
                        if a.lower() in exam_lower or exam_lower in a.lower():
                            matched_auxiliary = a
                            break

                    if matched_auxiliary:
                        if matched_auxiliary not in auxiliary_exams:
                            auxiliary_exams.append(matched_auxiliary)
                            print(f"    [DiagnosisAgent] 辅助检查规范化: '{exam[:30]}...' -> '{matched_auxiliary}'")
                    else:
                        print(f"    [DiagnosisAgent] 无法匹配检查项，抛弃: '{exam[:50]}...'")

        return {
            "physical_exams": physical_exams,
            "auxiliary_exams": auxiliary_exams
        }
