"""
Imaging Agent - 影像Agent
负责Task3: 根据辅助检查需求进行影像分析
"""
import os
from typing import Dict, List, Optional
from utils import load_prompt, img_api


# 需要影像分析的检查类型
IMAGING_TYPES = ['X-ray', 'CT', 'MRI', '超声', '病理检查', '内镜检查', '核医学成像']


class ImagingAgent:
    """影像Agent"""

    def __init__(self, img_base_dir: str = './datasets/MedImg/'):
        self.prompt_template = load_prompt("imaging")
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if os.path.isabs(img_base_dir):
            self.img_base_dir = os.path.normpath(img_base_dir)
        else:
            self.img_base_dir = os.path.normpath(os.path.join(project_root, img_base_dir))

    def run(self, auxiliary_exams: List[str], chief_complaint: str,
            images_info: List[Dict], gt_img_paths: List[str],
            gt_auxiliary_exam: Dict, case_id: str = "未知",
            completed_imaging_exams: List[str] = None) -> Dict:
        """
        执行影像分析任务

        Args:
            auxiliary_exams: Task2给出的辅助检查列表
            chief_complaint: 主诉
            images_info: 图像信息列表（包含分类和文件名）
            gt_img_paths: ground truth中的图像路径列表
            gt_auxiliary_exam: ground truth中的辅助检查结果
            completed_imaging_exams: 已完成的影像检查（用于回溯时过滤）

        Returns:
            {
                "imaging_results": {影像类型: 分析结果},
                "non_imaging_results": {非影像类检查: ground truth结果},
                "raw_outputs": {影像类型: 原始输出}
            }
        """
        imaging_results = {}
        non_imaging_results = {}
        raw_outputs = {}
        imaging_inputs = {}

        # 过滤已完成的检查
        if completed_imaging_exams:
            auxiliary_exams = [e for e in auxiliary_exams if e not in completed_imaging_exams]

        # 获取ground_truth中实际存在的检查项目
        gt_exam_keys = set(gt_auxiliary_exam.keys()) if gt_auxiliary_exam else set()
        print(f"    [ImagingAgent] Ground truth中存在的检查项目: {gt_exam_keys}")

        for exam in auxiliary_exams:
            # 判断是否是影像类检查
            is_imaging = self._is_imaging_exam(exam)
            imaging_type = self._get_imaging_type(exam) if is_imaging else None

            # 核心逻辑：只处理ground_truth中实际存在的检查项目
            # 通过模糊匹配检查该exam是否在gt中存在
            gt_matched_key = self._find_matching_gt_key(exam, gt_exam_keys)

            if not gt_matched_key:
                # ground_truth中没有该检查，跳过
                print(f"    [ImagingAgent] 跳过 '{exam}'：ground truth中不存在该检查")
                continue

            if is_imaging:
                # 优先从images_info获取图像路径
                img_paths = self._get_image_paths_for_exam(exam, images_info)

                # 如果images_info没有匹配到，检查gt_img_paths中是否有对应该检查类型的图像
                if not img_paths and gt_img_paths:
                    # 只有当gt中确实有这个检查类型的结果时，才使用gt_img_paths
                    img_paths = gt_img_paths
                    print(f"    [ImagingAgent] 使用gt_img_paths，共 {len(img_paths)} 张图像")

                if img_paths:
                    # 调用多模态LLM进行影像分析
                    prompt = self.prompt_template.format(
                        case_id=case_id,
                        chief_complaint=chief_complaint,
                        imaging_type=imaging_type
                    )

                    print(f"    [ImagingAgent] 调用多模态LLM分析 {exam} -> 规范化为 '{imaging_type}'，图像数量: {len(img_paths)}")
                    result = img_api(img_paths, prompt)
                    # 使用规范化的imaging_type作为key
                    imaging_results[imaging_type] = result
                    raw_outputs[imaging_type] = result
                    imaging_inputs[imaging_type] = {
                        "exam": exam,
                        "image_paths": img_paths
                    }
                else:
                    # 没有找到对应图像，使用ground truth文本结果
                    print(f"    [ImagingAgent] 未找到 {exam} 对应图像，使用ground truth文本结果，规范化为 '{imaging_type}'")
                    imaging_results[imaging_type] = gt_auxiliary_exam.get(gt_matched_key, "")
                    imaging_inputs[imaging_type] = {
                        "exam": exam,
                        "image_paths": []
                    }
            else:
                # 非影像类检查，直接使用ground truth
                normalized_exam = self._normalize_non_imaging_exam(exam)
                print(f"    [ImagingAgent] 非影像检查: {exam} -> 规范化为 '{normalized_exam}'")
                non_imaging_results[normalized_exam] = gt_auxiliary_exam.get(gt_matched_key, "")

        return {
            "imaging_results": imaging_results,
            "non_imaging_results": non_imaging_results,
            "raw_outputs": raw_outputs,
            "imaging_inputs": imaging_inputs
        }

    def _is_imaging_exam(self, exam: str) -> bool:
        """判断是否是影像类检查"""
        exam_lower = exam.lower()
        for img_type in IMAGING_TYPES:
            if img_type.lower() in exam_lower or exam_lower in img_type.lower():
                return True
        return False

    def _get_imaging_type(self, exam: str) -> str:
        """获取影像类型"""
        exam_lower = exam.lower()
        for img_type in IMAGING_TYPES:
            if img_type.lower() in exam_lower or exam_lower in img_type.lower():
                return img_type
        return exam

    def _normalize_non_imaging_exam(self, exam: str) -> str:
        """规范化非影像类检查名称"""
        # 非影像类检查的标准名称
        NON_IMAGING_EXAMS = ['血液学检查', '尿液检查', '粪便检查']

        exam_lower = exam.lower()
        for standard in NON_IMAGING_EXAMS:
            standard_lower = standard.lower()
            if standard_lower in exam_lower or exam_lower in standard_lower:
                return standard

        # 特殊匹配规则
        if '血' in exam or 'blood' in exam_lower:
            return '血液学检查'
        if '尿' in exam or 'urine' in exam_lower:
            return '尿液检查'
        if '粪' in exam or '便' in exam or 'stool' in exam_lower or 'fecal' in exam_lower:
            return '粪便检查'

        return exam

    def _find_matching_gt_key(self, exam: str, gt_keys: set) -> Optional[str]:
        """
        在ground_truth的key中查找与exam匹配的项

        Args:
            exam: 预测的检查名称
            gt_keys: ground_truth中存在的检查key集合

        Returns:
            匹配到的gt key，如果没有匹配则返回None
        """
        if not gt_keys:
            return None

        exam_lower = exam.lower()

        # 1. 精确匹配
        if exam in gt_keys:
            return exam

        # 2. 包含匹配
        for gt_key in gt_keys:
            gt_key_lower = gt_key.lower()
            # 双向包含匹配
            if gt_key_lower in exam_lower or exam_lower in gt_key_lower:
                return gt_key

        # 3. 关键词匹配（针对影像类型）
        exam_type = self._get_imaging_type(exam)
        if exam_type != exam:  # 说明匹配到了标准影像类型
            for gt_key in gt_keys:
                if exam_type.lower() in gt_key.lower():
                    return gt_key

        return None

    def _get_image_paths_for_exam(self, exam: str, images_info: List[Dict]) -> List[str]:
        """
        根据检查类型获取对应的图像路径

        Args:
            exam: 检查类型
            images_info: 图像信息列表，格式为 [{"分类": "MRI", "文件名": "xxx.jpg"}, ...]
                        或者 [{"type": "CT", "file": "xxx.jpg"}, ...]
                        或者直接是文件路径字符串列表

        Returns:
            图像路径列表
        """
        paths = []
        exam_lower = exam.lower()

        # 调试输出
        if images_info:
            print(f"    [ImagingAgent] 查找 {exam} 图像，images_info: {len(images_info)} 项")

        for img_info in images_info:
            if isinstance(img_info, dict):
                # 支持多种字段名
                category_raw = (
                    img_info.get('分类', '')
                    or img_info.get('category', '')
                    or img_info.get('type', '')
                    or img_info.get('类型', '')
                )
                filename = (
                    img_info.get('文件名', '')
                    or img_info.get('filename', '')
                    or img_info.get('file', '')
                    or img_info.get('path', '')
                )

                # 处理分类字段：可能是字符串或列表
                if isinstance(category_raw, list):
                    categories = category_raw  # 是列表，如 ["内镜检查"]
                elif isinstance(category_raw, str) and category_raw:
                    categories = [category_raw]  # 转为列表
                else:
                    categories = []

                # 匹配分类
                if filename:
                    matched = False
                    if categories:
                        for cat in categories:
                            cat_lower = cat.lower() if isinstance(cat, str) else ""
                            if (exam_lower in cat_lower or cat_lower in exam_lower or
                                    self._fuzzy_match_imaging_type(exam, cat)):
                                matched = True
                                break
                    else:
                        # 如果没有分类信息，尝试从文件名匹配
                        filename_lower = filename.lower()
                        if self._fuzzy_match_imaging_type(exam, filename_lower):
                            matched = True

                    if matched:
                        full_path = filename if os.path.isabs(filename) else os.path.join(self.img_base_dir, filename)
                        if os.path.exists(full_path):
                            paths.append(full_path)
                            print(f"    [ImagingAgent] 匹配到图像: {filename}")
                        else:
                            print(f"    [ImagingAgent] 警告：图像文件不存在: {full_path}")

            elif isinstance(img_info, str):
                # 直接是文件路径
                img_info = img_info.strip()
                full_path = os.path.join(self.img_base_dir, img_info) if not os.path.isabs(img_info) else img_info
                if os.path.exists(full_path):
                    # 根据文件名判断是否匹配
                    if self._fuzzy_match_imaging_type(exam, img_info.lower()):
                        paths.append(full_path)

        return paths

    def _fuzzy_match_imaging_type(self, exam: str, category: str) -> bool:
        """模糊匹配影像类型"""
        # 定义匹配规则
        match_rules = {
            'x-ray': ['x线', 'xray', 'x光', 'dr', 'cr'],
            'ct': ['ct', '电脑断层'],
            'mri': ['mri', '核磁', '磁共振'],
            '超声': ['超声', 'b超', '彩超', 'ultrasound'],
            '病理检查': ['病理', 'pathology'],
            '内镜检查': ['内镜', '胃镜', '肠镜', 'endoscopy'],
            '核医学成像': ['核医学', 'pet', 'spect', 'nuclear']
        }

        exam_lower = exam.lower()
        category_lower = category.lower()

        for key, aliases in match_rules.items():
            key_lower = key.lower()
            if key_lower in exam_lower or exam_lower in key_lower:
                for alias in aliases:
                    if alias in category_lower:
                        return True
            if key_lower in category_lower or category_lower in key_lower:
                for alias in aliases:
                    if alias in exam_lower:
                        return True

        return False

    def _fuzzy_match_exam(self, exam: str, gt_exams: Dict) -> Optional[str]:
        """模糊匹配检查项目"""
        exam_lower = exam.lower().replace('-', '').replace('_', '').replace(' ', '')

        for gt_key, gt_value in gt_exams.items():
            gt_lower = gt_key.lower().replace('-', '').replace('_', '').replace(' ', '')
            if exam_lower in gt_lower or gt_lower in exam_lower:
                return gt_value

        return None
