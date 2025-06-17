import os
from typing import List, Dict, Literal, Union
import cv2
import numpy as np

class ZoneMaskReader:
    DEFAULT_PATH = "D:/研究生实验/Fast_Online_MTMCT/Fast_Online_MTMCT/2. online_MTMC/preliminary/devied_zones/AIC19_2"
    
    def __init__(self, root_path: str = DEFAULT_PATH):
        """
        初始化区域mask读取器
        :param root_path: 根目录路径
        """
        self.root_path = root_path
        self.zone_masks: Dict[str, Dict[int, Dict[str, np.ndarray]]] = {}
        self._masks_loaded = False

    def ensure_masks_loaded(self):
        """确保mask已加载"""
        if not self._masks_loaded:
            self.load_zone_masks()
            self._masks_loaded = True

    def load_zone_masks(self):
        """
        加载所有区域的mask图像
        格式: self.zone_masks = {
            "c006": {
                1: {"entry": mask, "exit": mask, "middle": mask},
                2: {"entry": mask, "exit": mask, "middle": mask},
                ...
                9: {"entry": mask, "exit": mask, "middle": mask}
            },
            ...
        }
        """
        # 初始化摄像头字典
        for cam_num in range(6, 10):
            cam_id = f"c{cam_num:03d}"
            self.zone_masks[cam_id] = {}
            
            # 遍历1-9号车道
            for lane_num in range(1, 10):
                self.zone_masks[cam_id][lane_num] = {
                    'entry': None,
                    'exit': None,
                    'middle': None
                }
                
                # 构建路径：root_path/lane_num/cam_num.jpg
                mask_path = os.path.join(self.root_path, str(lane_num), f"{cam_num}.jpg")
                if os.path.exists(mask_path):
                    # 读取彩色图像
                    img = cv2.imread(mask_path)
                    if img is None:
                        print(f"Warning: Failed to read image at {mask_path}")
                        continue

                    # 分别提取红、蓝、白色区域的mask
                    height, width = img.shape[:2]
                    
                    # BGR格式
                    # 红色区域 (入口)
                    red_mask = cv2.inRange(img, np.array([0, 0, 150]), np.array([80, 80, 255]))
                    if np.any(red_mask):
                        self.zone_masks[cam_id][lane_num]['entry'] = red_mask
                    
                    # 蓝色区域 (出口)
                    blue_mask = cv2.inRange(img, np.array([150, 0, 0]), np.array([255, 80, 80]))
                    if np.any(blue_mask):
                        self.zone_masks[cam_id][lane_num]['exit'] = blue_mask
                    
                    # 白色区域 (中间)
                    white_mask = cv2.inRange(img, np.array([240, 240, 240]), np.array([255, 255, 255]))
                    if np.any(white_mask):
                        self.zone_masks[cam_id][lane_num]['middle'] = white_mask
                else:
                    print(f"Warning: Mask not found at {mask_path}")

    def get_zone_type(self, bbox: List[int], cam_id: str, lane_num: Union[int, str]) -> List[Literal['entry', 'exit', 'middle']]:
        """
        判断给定bbox在哪个区域内
        :param bbox: [x1, y1, x2, y2] 格式的bbox坐标
        :param cam_id: 摄像头ID (c006-c009)
        :param lane_num: 车道号 (1-9)，可以是整数或字符串
        :return: 包含区域类型的列表
        """
        self.ensure_masks_loaded()
        
        # 将车道号转换为整数
        try:
            lane_num = int(lane_num)
        except (ValueError, TypeError):
            print(f"无效的车道号格式: {lane_num}")
            return []

        if cam_id not in self.zone_masks or lane_num not in self.zone_masks[cam_id]:
            print(f"无效的摄像头ID或车道号: {cam_id}, {lane_num}")
            return []

        x1, y1, x2, y2 = bbox
        bbox_width = x2 - x1
        bbox_height = y2 - y1
        bbox_area = bbox_width * bbox_height

        detected_zones = []
        
        # 遍历该摄像头该车道下的所有区域mask
        for zone_type, mask in self.zone_masks[cam_id][lane_num].items():
            if mask is None or mask.shape[0] == 0 or mask.shape[1] == 0:
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
            
            # 如果白色像素占比超过阈值，则认为在该区域内
            if white_ratio > 0.2:
                detected_zones.append(zone_type)
        
        return detected_zones

    def debug_bbox_mask(self, bbox: List[int], cam_id: str, lane_num: int, output_dir: str = "debug_output"):
        """
        将bbox区域的mask信息保存为图片用于debug
        :param bbox: [x1, y1, x2, y2] 格式的bbox坐标
        :param cam_id: 摄像头ID (c006-c009)
        :param lane_num: 车道号 (1-9)
        :param output_dir: 输出目录
        """
        self.ensure_masks_loaded()
        if cam_id not in self.zone_masks or lane_num not in self.zone_masks[cam_id]:
            print(f"无效的摄像头ID或车道号: {cam_id}, {lane_num}")
            return

        os.makedirs(output_dir, exist_ok=True)
        
        x1, y1, x2, y2 = bbox
        
        # 遍历该摄像头该车道下的所有区域mask
        for zone_type, mask in self.zone_masks[cam_id][lane_num].items():
            if mask is None or mask.shape[0] == 0 or mask.shape[1] == 0:
                continue

            debug_img = cv2.cvtColor(mask.copy(), cv2.COLOR_GRAY2BGR)
            
            x1_safe = max(0, x1)
            y1_safe = max(0, y1)
            x2_safe = min(mask.shape[1], x2)
            y2_safe = min(mask.shape[0], y2)
            
            cv2.rectangle(debug_img, (x1_safe, y1_safe), (x2_safe, y2_safe), (0, 255, 0), 2)
            
            roi = mask[y1_safe:y2_safe, x1_safe:x2_safe]
            if roi.size == 0:
                continue
            
            bbox_area = (x2 - x1) * (y2 - y1)
            white_pixels = np.sum(roi == 255)
            white_ratio = white_pixels / bbox_area
            
            text = f"Lane: {lane_num}, Zone: {zone_type}, White ratio: {white_ratio:.2f}"
            cv2.putText(debug_img, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                        1, (0, 0, 255), 2)
            
            output_path = os.path.join(output_dir, f"{cam_id}_lane{lane_num}_{zone_type}_debug.png")
            cv2.imwrite(output_path, debug_img)
            
            roi_colored = cv2.cvtColor(roi, cv2.COLOR_GRAY2BGR)
            roi_output_path = os.path.join(output_dir, f"{cam_id}_lane{lane_num}_{zone_type}_roi.png")
            cv2.imwrite(roi_output_path, roi_colored)

# 创建全局实例
zone_mask_reader = ZoneMaskReader()

if __name__ == '__main__':
    # 测试代码
    zone_reader = ZoneMaskReader()
    
    test_bbox = [374, 373, 374+249, 373+219]
    test_cam_id = 'c006'
    test_lane = 1

    zones = zone_reader.get_zone_type(test_bbox, test_cam_id, test_lane)
    print(f"Camera {test_cam_id}, bbox {test_bbox} 所在区域:", zones)
    
    # Debug输出
    zone_reader.debug_bbox_mask(test_bbox, test_cam_id, test_lane)