import os
from typing import List, Dict
import cv2
import numpy as np

class LaneMaskReader:
    DEFAULT_PATH = "D:/研究生实验/Fast_Online_MTMCT/Fast_Online_MTMCT/2. online_MTMC/preliminary/devied_zones/AIC19_compressed"
    
    def __init__(self, root_path: str = DEFAULT_PATH):
        """
        初始化车道mask读取器
        :param root_path: 根目录路径，默认使用AIC19数据集路径
        """
        self.root_path = root_path
        # 改为懒加载，初始化时只创建空字典
        self.lane_masks: Dict[str, Dict[str, np.ndarray]] = {}
        self._masks_loaded = False

    def ensure_masks_loaded(self):
        """确保mask已加载"""
        if not self._masks_loaded:
            self.load_lane_masks()
            self._masks_loaded = True

    def get_subfolder_names(self) -> List[int]:
        """
        获取子文件夹名称列表
        :return: 子文件夹名称列表
        """
        try:
            # 使用os.listdir获取目录下所有内容
            items = os.listdir(self.root_path)
            # 过滤出文件夹
            subfolders = sorted([item for item in items
                         if os.path.isdir(os.path.join(self.root_path, item))])
            return subfolders
        except Exception as e:
            print(f"读取文件夹出错: {str(e)}")
            return []

    def get_subfolder_paths(self) -> List[str]:
        """
        获取子文件夹的完整路径列表
        :return: 子文件夹的完整路径列表
        """
        subfolders = self.get_subfolder_names()
        return [os.path.join(self.root_path, folder) for folder in subfolders]

    def load_lane_masks(self):
        """
        加载所有车道的mask图像
        格式: self.lane_masks = {
            "c006": {"1": mask1, "2": mask2, ..., "9": mask9},
            "c007": {"1": mask1, "2": mask2, ..., "9": mask9},
            ...
        }
        """
        # 初始化摄像头字典
        for cam_num in range(6, 10):
            cam_id = f"c{cam_num:03d}"
            self.lane_masks[cam_id] = {}

        subfolder_names = self.get_subfolder_names()
        for lane_name in subfolder_names:
            lane_path = os.path.join(self.root_path, lane_name)
            
            # 遍历每个摄像头
            for cam_num in range(6, 10):
                cam_id = f"c{cam_num:03d}"
                mask_path = os.path.join(lane_path, f"{cam_num}.jpg")
                if os.path.exists(mask_path):
                    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                    # 二值化，确保白色区域为255
                    _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
                    self.lane_masks[cam_id][lane_name] = mask

    def get_lanes_for_bbox(self, bbox: List[int], cam_id: str) -> List[str]:
        """
        判断给定bbox在哪些车道上
        :param bbox: [x1, y1, x2, y2] 格式的bbox坐标
        :param cam_id: 摄像头ID (c006-c009)
        :return: 包含车道号的列表
        """
        self.ensure_masks_loaded()  # 确保mask已加载
        if cam_id not in self.lane_masks:
            print(f"无效的摄像头ID: {cam_id}")
            return []

        x1, y1, x2, y2 = bbox
        bbox_width = x2 - x1
        bbox_height = y2 - y1
        bbox_area = bbox_width * bbox_height

        detected_lanes = []
        
        # 遍历该摄像头下的所有车道mask
        for lane_name, mask in self.lane_masks[cam_id].items():
            if mask.shape[0] == 0 or mask.shape[1] == 0:
                continue

            # 确保bbox在图像范围内
            x1_safe = max(0, x1)
            y1_safe = max(0, y1)
            x2_safe = min(mask.shape[1], x2)
            y2_safe = min(mask.shape[0], y2)
            
            # 提取bbox区域的mask
            roi = mask[y1_safe:y2_safe, x1_safe:x2_safe]
            if roi.size == 0:
                continue
            
            # 计算白色像素（255）占比
            white_pixels = np.sum(roi == 255)
            white_ratio = white_pixels / bbox_area
            
            # 如果白色像素占比超过50%，则认为在该车道上
            if white_ratio > 0.3:
                detected_lanes.append(lane_name)
        
        return detected_lanes

    def debug_bbox_mask(self, bbox: List[int], cam_id: str, output_dir: str = "debug_output"):
        """
        将bbox区域的mask信息保存为图片用于debug
        :param bbox: [x1, y1, x2, y2] 格式的bbox坐标
        :param cam_id: 摄像头ID (c006-c009)
        :param output_dir: 输出目录
        """
        self.ensure_masks_loaded()  # 确保mask已加载
        if cam_id not in self.lane_masks:
            print(f"无效的摄像头ID: {cam_id}")
            return

        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        x1, y1, x2, y2 = bbox
        
        # 遍历该摄像头下的所有车道mask
        for lane_name, mask in self.lane_masks[cam_id].items():
            if mask.shape[0] == 0 or mask.shape[1] == 0:
                continue

            # 创建彩色图像用于可视化
            debug_img = cv2.cvtColor(mask.copy(), cv2.COLOR_GRAY2BGR)
            
            # 确保bbox在图像范围内
            x1_safe = max(0, x1)
            y1_safe = max(0, y1)
            x2_safe = min(mask.shape[1], x2)
            y2_safe = min(mask.shape[0], y2)
            
            # 在原图上画出bbox
            cv2.rectangle(debug_img, (x1_safe, y1_safe), (x2_safe, y2_safe), (0, 255, 0), 2)
            
            # 提取bbox区域的mask
            roi = mask[y1_safe:y2_safe, x1_safe:x2_safe]
            if roi.size == 0:
                continue
            
            # 计算白色像素占比
            bbox_area = (x2 - x1) * (y2 - y1)
            white_pixels = np.sum(roi == 255)
            white_ratio = white_pixels / bbox_area
            
            # 在图片上添加信息
            text = f"Lane: {lane_name}, White ratio: {white_ratio:.2f}"
            cv2.putText(debug_img, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                        1, (0, 0, 255), 2)
            
            # 保存debug图片
            output_path = os.path.join(output_dir, f"{cam_id}_lane{lane_name}_debug.png")
            cv2.imwrite(output_path, debug_img)
            
            # 单独保存ROI区域
            roi_colored = cv2.cvtColor(roi, cv2.COLOR_GRAY2BGR)
            roi_output_path = os.path.join(output_dir, f"{cam_id}_lane{lane_name}_roi.png")
            cv2.imwrite(roi_output_path, roi_colored)

lane_mask_reader = LaneMaskReader()

if __name__ == '__main__':

    # 创建实例 (使用默认路径)
    lane_reader = LaneMaskReader()


    # 测试bbox检测
    test_bbox = [374, 373, 374+249, 373 + 219]  # [x1, y1, x2, y2]
    test_cam_id = 'c006'

    lanes = lane_reader.get_lanes_for_bbox(test_bbox, test_cam_id)
    print(f"Camera {test_cam_id}, bbox {test_bbox} 所在车道:", lanes)

    # 在原有代码的测试部分添加：
    # lane_reader.debug_bbox_mask(test_bbox, test_cam_id)
