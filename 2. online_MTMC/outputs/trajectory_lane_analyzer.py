import os
import pandas as pd
import numpy as np
from typing import Dict, List, Set, Tuple
from collections import defaultdict, Counter
from utils.laneUtils import LaneMaskReader
from tqdm import tqdm
from tabulate import tabulate

class TrajectoryLaneAnalyzer:
    def __init__(self, gt_path: str, lane_mask_reader: LaneMaskReader = None):
        """
        初始化轨迹车道分析器
        :param gt_path: GT文件路径
        :param lane_mask_reader: 车道mask读取器实例
        """
        self.gt_path = gt_path
        self.lane_reader = lane_mask_reader or LaneMaskReader()
        self.trajectories = self._load_gt_file()
        # {track_id: {cam_id: {frame_id: set(lanes)}}}
        self.track_frame_lanes = defaultdict(lambda: defaultdict(dict))
        # {track_id: {cam_id: {lane_id: confidence}}}
        self.track_lane_confidence = defaultdict(lambda: defaultdict(dict))
        
    def _load_gt_file(self) -> pd.DataFrame:
        """
        加载GT文件
        格式: CameraId,Id,FrameId,X,Y,Width,Height,Xworld,Yworld
        支持txt格式，支持多种分隔符
        """
        try:
            # 定义列名
            names = ['camera_id', 'track_id', 'frame_id', 'x1', 'y1', 'w', 'h', 'xworld', 'yworld']
            
            # 读取文件
            df = pd.read_csv(
                self.gt_path,
                sep='\s+|\t+|,',  # 支持空格、制表符、逗号分隔
                index_col=None,
                skipinitialspace=True,
                header=None,
                names=names,
                engine='python'
            )
            
            print(f"读取到的数据形状: {df.shape}")  # 调试信息
            print(f"数据预览:\n{df.head()}")  # 调试信息
            
            # 确保数据类型正确
            df['frame_id'] = df['frame_id'].astype(int)
            df['track_id'] = df['track_id'].astype(int)
            df['camera_id'] = df['camera_id'].astype(int)
            df[['x1', 'y1', 'w', 'h']] = df[['x1', 'y1', 'w', 'h']].astype(float)
            
            return df
            
        except Exception as e:
            print(f"加载GT文件失败: {str(e)}")
            raise e

    def analyze_trajectories(self):
        """分析所有轨迹的车道占用情况"""
        if self.trajectories.empty:
            print("没有可分析的轨迹数据")
            return

        print("开始分析轨迹...")
        
        # 1. 收集每帧的车道信息
        print("步骤1/3: 收集帧级车道信息")
        self._collect_frame_lanes()
        
        # 2. 分析每个轨迹的车道置信度
        print("步骤2/3: 分析车道置信度")
        self._analyze_lane_confidence()
        
        # 3. 应用车道连续性约束
        print("步骤3/3: 应用连续性约束")
        self._apply_continuity_constraint()

    def _collect_frame_lanes(self):
        """收集每帧的车道信息"""
        # 使用tqdm包装groupby对象
        for track_id, track_data in tqdm(self.trajectories.groupby('track_id'), desc="处理轨迹"):
            for camera_id, cam_data in track_data.groupby('camera_id'):
                cam_id = f"c{camera_id:03d}"
                
                for _, row in cam_data.iterrows():
                    bbox = [
                        int(row['x1']),
                        int(row['y1']),
                        int(row['x1'] + row['w']),
                        int(row['y1'] + row['h'])
                    ]
                    frame_id = row['frame_id']
                    
                    lanes = self.lane_reader.get_lanes_for_bbox(bbox, cam_id)
                    if lanes:
                        self.track_frame_lanes[track_id][cam_id][frame_id] = set(lanes)

    def _analyze_lane_confidence(self):
        """分析每个轨迹在每个摄像头下的车道置信度"""
        for track_id, cam_data in tqdm(self.track_frame_lanes.items(), desc="计算置信度"):
            for cam_id, frame_data in cam_data.items():
                all_lanes = []
                for lanes in frame_data.values():
                    all_lanes.extend(list(lanes))
                
                if not all_lanes:
                    continue
                    
                lane_counter = Counter(all_lanes)
                total_frames = len(frame_data)
                
                for lane, count in lane_counter.items():
                    confidence = count / total_frames
                    self.track_lane_confidence[track_id][cam_id][lane] = confidence

    def _apply_continuity_constraint(self, window_size: int = 5):
        """应用车道连续性约束"""
        for track_id, cam_data in tqdm(self.track_frame_lanes.items(), desc="应用连续性约束"):
            for cam_id, frame_data in cam_data.items():
                frame_ids = sorted(frame_data.keys())
                
                for i in range(len(frame_ids)):
                    start_idx = max(0, i - window_size // 2)
                    end_idx = min(len(frame_ids), i + window_size // 2 + 1)
                    window_frames = frame_ids[start_idx:end_idx]
                    
                    window_lanes = []
                    for frame_id in window_frames:
                        window_lanes.extend(list(frame_data[frame_id]))
                    
                    if window_lanes:
                        most_common_lanes = Counter(window_lanes).most_common(1)
                        frame_data[frame_ids[i]] = {most_common_lanes[0][0]}

    def get_track_summary(self) -> Dict:
        """获取轨迹分析摘要"""
        summary = {
            'total_tracks': len(self.track_lane_confidence),
            'camera_stats': defaultdict(int),
            'track_details': {}
        }

        for track_id, cam_data in self.track_lane_confidence.items():
            track_info = {}
            for cam_id, lane_conf in cam_data.items():
                # 更新摄像头统计
                summary['camera_stats'][cam_id] += 1
                
                # 按置信度排序的车道列表
                sorted_lanes = sorted(
                    lane_conf.items(),
                    key=lambda x: x[1],
                    reverse=True
                )
                track_info[cam_id] = {
                    'primary_lane': sorted_lanes[0] if sorted_lanes else None,
                    'all_lanes': sorted_lanes
                }
            summary['track_details'][track_id] = track_info

        return summary

    def save_analysis_results(self, output_path: str):
        """保存分析结果到文件"""
        summary = self.get_track_summary()
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("轨迹车道分析报告\n")
            f.write("=" * 50 + "\n\n")
            
            # 基本统计信息
            stats_table = [
                ["总轨迹数", summary['total_tracks']],
                ["总摄像头数", len(summary['camera_stats'])]
            ]
            f.write(tabulate(stats_table, tablefmt="grid") + "\n\n")
            
            # 摄像头统计
            cam_stats = [[cam_id, count] for cam_id, count in summary['camera_stats'].items()]
            f.write("摄像头覆盖情况:\n")
            f.write(tabulate(cam_stats, headers=["摄像头", "轨迹数"], tablefmt="grid") + "\n\n")
            
            # 详细轨迹信息
            f.write("详细轨迹信息:\n")
            for track_id, track_info in tqdm(summary['track_details'].items(), desc="保存分析结果"):
                f.write(f"\n轨迹ID {track_id}:\n")
                
                # 收集该轨迹的所有信息用于表格显示
                track_table = []
                primary_lanes = {}
                
                for cam_id, cam_info in track_info.items():
                    if cam_info['primary_lane']:
                        lane, conf = cam_info['primary_lane']
                        primary_lanes[cam_id] = (lane, conf)
                        
                        # 主要车道信息
                        row = [cam_id, lane, f"{conf:.2f}"]
                        
                        # 其他可能车道信息
                        other_lanes = [
                            f"{l}({c:.2f})" for l, c in cam_info['all_lanes'][1:]
                            if c > 0.3
                        ]
                        row.append(", ".join(other_lanes) if other_lanes else "-")
                        
                        track_table.append(row)
                
                # 输出轨迹表格
                f.write(tabulate(
                    track_table,
                    headers=["摄像头", "主要车道", "置信度", "其他可能车道"],
                    tablefmt="grid"
                ) + "\n")
                
                # 跨摄像头分析
                if len(primary_lanes) > 1:
                    f.write("\n跨摄像头分析:\n")
                    common_lanes = set()
                    for cam_id, (lane, _) in primary_lanes.items():
                        if not common_lanes:
                            common_lanes = {lane}
                        else:
                            common_lanes &= {lane}
                    
                    cross_cam_table = [[
                        "共同车道",
                        ", ".join(map(str, common_lanes)) if common_lanes else "无"
                    ]]
                    f.write(tabulate(cross_cam_table, tablefmt="grid") + "\n")

def main():
    print("开始车道轨迹分析...")
    
    gt_path = "/root/mtmct/2. online_MTMC/outputs/ground_truth_validation.txt"
    analyzer = TrajectoryLaneAnalyzer(gt_path)
    
    # 分析轨迹
    analyzer.analyze_trajectories()
    
    # 保存分析结果
    output_file = "trajectory_lane_analysis.txt"
    print(f"\n保存分析结果到: {output_file}")
    analyzer.save_analysis_results(output_file)
    
    # 打印基本统计信息
    summary = analyzer.get_track_summary()
    print("\n分析完成！基本统计信息：")
    
    # 使用表格形式打印统计信息
    stats_table = [
        ["总轨迹数", summary['total_tracks']],
        ["总摄像头数", len(summary['camera_stats'])]
    ]
    print(tabulate(stats_table, tablefmt="grid"))
    
    print("\n摄像头覆盖情况:")
    cam_stats = [[cam_id, count] for cam_id, count in summary['camera_stats'].items()]
    print(tabulate(cam_stats, headers=["摄像头", "轨迹数"], tablefmt="grid"))

if __name__ == "__main__":
    main()