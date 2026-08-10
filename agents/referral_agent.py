"""
Referral Agent - 分诊Agent
负责Task1: 根据病人主诉进行分诊
"""

DEPT_L1_OPTIONS = [
    "内科", "外科", "肿瘤科", "妇产科", "医学影像科", "儿科", "耳鼻咽喉科", "口腔科",
    "皮肤性病科", "中医科", "康复科", "急诊科", "眼科", "护理科", "精神科", "全科", "检验科"
]

DEPT_L2_OPTIONS = {
    "内科": [
        "肾脏内科", "结核病科", "肝病科", "消化内科", "内分泌科", "神经内科", "传染科",
        "过敏反应科", "干部诊疗科", "呼吸科", "免疫科", "心血管内科", "血液科",
        "老年病科", "风湿科", "感染科"
    ],
    "外科": [
        "创伤骨科", "脊柱外科", "外伤科", "麻醉疼痛科", "骨肿瘤科", "泌尿外科",
        "胃肠外科", "神经外科", "心脏外科", "普外科", "乳腺外科", "关节骨科", "骨科",
        "血管外科", "肝胆外科", "手外科", "心胸外科", "整形科", "烧伤科", "胸外科",
        "肛肠外科", "微创外科"
    ],
    "肿瘤科": ["肿瘤妇科", "肿瘤外科", "放疗科", "肿瘤内科"],
    "妇产科": ["生殖中心", "产前检查科", "妇科肿瘤", "计划生育科", "高危产科", "产科", "妇科"],
    "医学影像科": ["MRI室", "CT室", "B超科", "X线室", "彩超科", "放射科", "超声科", "核医学科"],
    "儿科": [
        "小儿感染科", "小儿耳鼻喉", "小儿免疫科", "小儿血液科", "小儿心内科", "小儿呼吸科",
        "小儿骨科", "小儿内分泌科", "小儿消化科", "小儿精神科", "小儿神经外科",
        "小儿外科", "小儿心内科", "小儿神经内科", "小儿急诊科"
    ],
    "耳鼻咽喉科": ["耳鼻咽喉科"],
    "口腔科": ["牙周科", "儿童口腔科", "口腔修复科", "牙体牙髓科", "种植科", "正畸科", "口腔预防科", "颌面外科"],
    "皮肤性病科": ["激光室", "性病科", "皮肤美容", "皮肤科"],
    "中医科": ["中医消化科", "中医内分泌", "中西医结合科", "中医免疫内科", "中医老年病科", "中医呼吸科", "中医内科", "针灸科"],
    "康复科": ["康复科"],
    "急诊科": ["急诊科"],
    "眼科": ["眼外伤", "青光眼", "眼眶及肿瘤", "眼视光学", "角膜科", "白内障", "小儿眼科", "眼底"],
    "护理科": ["基础护理", "内科护理"],
    "精神科": ["精神科"],
    "全科": ["全科"],
    "检验科": ["血液检验"]
}

import re
from typing import Dict, List, Tuple
from utils import load_prompt, call_llm_api, parse_json_from_response


class ReferralAgent:
    """分诊Agent"""

    def __init__(self):
        self.l1_prompt_template = load_prompt("referral_l1")
        self.l2_prompt_template = load_prompt("referral_l2")

    @staticmethod
    def _extract_single_choice(response: str, options: List[str]) -> str:
        if not response:
            return "None"
        for line in response.splitlines():
            candidate = line.strip()
            if candidate in options:
                return candidate
        for option in options:
            if option in response:
                return option
        parts = re.split(r'[，,、\n]', response)
        parts = [p.strip() for p in parts if p.strip()]
        return parts[0] if parts else "None"

    @staticmethod
    def _extract_multi_choices(response: str, options: List[str]) -> List[str]:
        if not response:
            return []
        parts = re.split(r'[，,、\n]', response)
        candidates = [p.strip() for p in parts if p.strip()]
        selected = [c for c in candidates if c in options]
        if selected:
            return selected
        fallback = [option for option in options if option in response]
        return fallback

    def run_l1(self, chief_complaint: str, present_illness: str, case_id: str = "未知") -> Dict:
        prompt = self.l1_prompt_template.format(
            case_id=case_id,
            chief_complaint=chief_complaint,
            present_illness=present_illness,
            dept_l1_options="、".join(DEPT_L1_OPTIONS)
        )

        response = call_llm_api("你是一名分诊医生。", prompt)
        dept_l1 = self._extract_single_choice(response, DEPT_L1_OPTIONS)

        return {
            "dept_l1": dept_l1,
            "dept_l2": [],
            "raw_output": response
        }

    def run_l2(self, chief_complaint: str, present_illness: str, dept_l1: str, case_id: str = "未知") -> Dict:
        dept_l2_options = DEPT_L2_OPTIONS.get(dept_l1, [])
        prompt = self.l2_prompt_template.format(
            case_id=case_id,
            chief_complaint=chief_complaint,
            present_illness=present_illness,
            dept_l1=dept_l1,
            dept_l2_options="、".join(dept_l2_options) if dept_l2_options else "无"
        )

        response = call_llm_api("你是一名分诊医生。", prompt)
        dept_l2 = self._extract_multi_choices(response, dept_l2_options)

        return {
            "dept_l1": dept_l1,
            "dept_l2": dept_l2,
            "raw_output": response
        }

    def run(self, chief_complaint: str, present_illness: str, case_id: str = "未知") -> Dict:
        """
        执行分诊任务

        Args:
            chief_complaint: 病人主诉

        Returns:
            {
                "dept_l1": 一级科室,
                "dept_l2": [二级科室列表],
                "raw_output": 原始输出
            }
        """
        l1_result = self.run_l1(chief_complaint, present_illness, case_id=case_id)
        dept_l1 = l1_result["dept_l1"]
        l2_result = self.run_l2(chief_complaint, present_illness, dept_l1, case_id=case_id)
        l2_result["raw_output"] = f"{l1_result['raw_output']}\n{l2_result['raw_output']}"
        return l2_result


class ReferralVerifier:
    """分诊结果验证器"""

    def __init__(self):
        self.verify_prompt = load_prompt("referral_verify")
        self.final_prompt = load_prompt("referral_final")

    def verify(
        self,
        personal_info: Dict,
        chief_complaint: str,
        present_illness: str,
        dept_l1: str,
        dept_l2: List[str],
        case_id: str = "未知"
    ) -> Dict:
        """
        验证分诊结果

        Returns:
            {
                "dept_l1_correct": bool,
                "dept_l2_correct": bool,
                "reason": str,
                "suggested_dept_l1": str or None,
                "suggested_dept_l2": str or None
            }
        """
        prompt = self.verify_prompt.format(
            case_id=case_id,
            personal_info=str(personal_info),
            chief_complaint=chief_complaint,
            present_illness=present_illness,
            dept_l1=dept_l1,
            dept_l2=", ".join(dept_l2) if dept_l2 else "无"
        )

        response = call_llm_api("你是一名资深分诊专家。", prompt)
        result = parse_json_from_response(response)

        if result:
            return result
        else:
            # 如果解析失败，默认认为正确
            return {
                "dept_l1_correct": True,
                "dept_l2_correct": True,
                "reason": "",
                "suggested_dept_l1": None,
                "suggested_dept_l2": None
            }

    def get_final_result(
        self,
        personal_info: Dict,
        chief_complaint: str,
        present_illness: str,
        agent_dept_l1: str,
        agent_dept_l2: List[str],
        verify_result: Dict,
        case_id: str = "未知"
    ) -> Dict:
        """
        获取最终分诊结果

        Returns:
            {
                "final_dept_l1": str,
                "final_dept_l2": list,
                "confidence": int,
                "reasoning": str
            }
        """
        prompt = self.final_prompt.format(
            case_id=case_id,
            personal_info=str(personal_info),
            chief_complaint=chief_complaint,
            present_illness=present_illness,
            agent_dept_l1=agent_dept_l1,
            agent_dept_l2=", ".join(agent_dept_l2) if agent_dept_l2 else "无",
            verify_result=str(verify_result)
        )

        response = call_llm_api("你是一名资深分诊专家。", prompt)
        result = parse_json_from_response(response)

        if result:
            return result
        else:
            # 解析失败时返回原始结果
            return {
                "final_dept_l1": agent_dept_l1,
                "final_dept_l2": agent_dept_l2,
                "confidence": 50,
                "reasoning": "无法解析最终结果，使用原始分诊结果"
            }
