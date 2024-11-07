import cv2
import pandas as pd
from pathlib import Path
from typing import Dict, List
from utils.laneUtils import LaneMaskReader

class TrajectoryLaneAnalyzer:
    def __init__(self, video_path: str, gt_path: str):
        """
        初始化轨迹车道分析器
        :param video_path: 视频文件路径
        :param gt_path: GT文件路径（txt格式）
        """
        self.video_path = video_path
        self.gt_path = gt_path
        self.lane_reader = LaneMaskReader()
        
        # 读取视频信息
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise ValueError(f"无法打开视频: {video_path}")
            
        # 设置摄像头ID
        self.cam_id = "c006"  # 根据实际情况修改
        
        # 定义列名
        names = ['CameraId', 'Id', 'FrameId', 'X', 'Y', 'Width', 'Height', 'Xworld', 'Yworld']
        
        # 读取txt文件
        try:
            self.gt_data = pd.read_csv(
                gt_path,
                sep='\s+|\t+|,',  # 支持多种分隔符
                index_col=None,
                skipinitialspace=True,
                header=None,
                names=names,
                engine='python'
            )
        except Exception as e:
            raise ValueError(f"Could not read input from {gt_path}. Error: {repr(e)}")
        
        # 只保留当前摄像头的数据
        self.gt_data = self.gt_data[self.gt_data['CameraId'] == int(self.cam_id[1:])]

    def analyze_trajectories(self) -> Dict[int, Dict[int, List[str]]]:
        """
        分析所有轨迹的车道情况
        :return: 字典格式：{track_id: {frame_id: [车道列表]}}
        """
        trajectory_lanes = {}
        
        # 按track_id分组处理
        for track_id, track_data in self.gt_data.groupby('Id'):
            trajectory_lanes[track_id] = {}
            
            # 处理每一帧的数据
            for _, row in track_data.iterrows():
                frame_id = int(row['FrameId'])
                # 构建bbox [x1, y1, x2, y2]
                bbox = [
                    int(row['X']),
                    int(row['Y']),
                    int(row['X'] + row['Width']),
                    int(row['Y'] + row['Height'])
                ]
                
                # 获取当前bbox所在的车道
                lanes = self.lane_reader.get_lanes_for_bbox(bbox, self.cam_id)
                trajectory_lanes[track_id][frame_id] = lanes
                
        return trajectory_lanes

    def visualize_trajectory_lanes(self, output_dir: str = "trajectory_debug"):
        """
        可视化轨迹车道分析结果，合并连续且车道相同的帧
        :param output_dir: 输出目录
        """
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # 分析轨迹
        trajectory_lanes = self.analyze_trajectories()
        
        # 准备输出文件
        output_file = Path(output_dir) / f"{self.cam_id}_trajectory_analysis.txt"
        
        with open(output_file, 'w', encoding='utf-8') as f:
            for track_id, frame_data in trajectory_lanes.items():
                f.write(f"\n轨迹ID: {track_id}\n")
                f.write("-" * 50 + "\n")
                
                # 统计该轨迹中出现的所有车道
                all_lanes = set()
                for lanes in frame_data.values():
                    all_lanes.update(lanes)
                
                f.write(f"经过的所有车道: {sorted(list(all_lanes))}\n\n")
                
                # 合并连续且车道相同的帧
                frame_ids = sorted(frame_data.keys())
                if not frame_ids:
                    continue
                    
                current_lanes = frame_data[frame_ids[0]]
                start_frame = frame_ids[0]
                prev_frame = frame_ids[0]
                
                for frame_id in frame_ids[1:]:
                    lanes = frame_data[frame_id]
                    
                    # 如果车道不同或者帧不连续，输出之前的结果并开始新的统计
                    if lanes != current_lanes or frame_id != prev_frame + 1:
                        if start_frame == prev_frame:
                            f.write(f"Frame {start_frame}: {current_lanes}\n")
                        else:
                            f.write(f"Frame {start_frame}-{prev_frame}: {current_lanes}\n")
                        start_frame = frame_id
                        current_lanes = lanes
                    
                    prev_frame = frame_id
                
                # 输出最后一段
                if start_frame == prev_frame:
                    f.write(f"Frame {start_frame}: {current_lanes}\n")
                else:
                    f.write(f"Frame {start_frame}-{prev_frame}: {current_lanes}\n")
                
                f.write("\n")

def main():
    # 示例用法
    video_path = "/home/chatmindai/project/zhangkun/Fast_Online_MTMCT/dataset/AIC19/validation/S02/c006/vdo.avi"
    gt_path = "/home/chatmindai/project/zhangkun/Fast_Online_MTMCT/2. online_MTMC/outputs/ground_truth_validation.txt"
    
    analyzer = TrajectoryLaneAnalyzer(video_path, gt_path)
    
    # 进行分析并可视化
    analyzer.visualize_trajectory_lanes()

if __name__ == "__main__":
    main()
