import cv2
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional
from utils.laneUtils import LaneMaskReader
import numpy as np
from tqdm import tqdm

class TrajectoryLaneAnalyzer:
    def __init__(self, video_path: str, gt_path: str, cam_id: str):
        """
        初始化轨迹车道分析器
        :param video_path: 视频文件路径
        :param gt_path: GT文件路径（txt格式）
        :param cam_id: 摄像头ID
        """
        self.video_path = video_path
        self.gt_path = gt_path
        self.lane_reader = LaneMaskReader()
        self.cam_id = cam_id  # 现在从参数接收摄像头ID
        
        # 读取视频信息
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise ValueError(f"无法打开视频: {video_path}")
            
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
        可视化轨迹车道分析结果，合并连续且车道相同的帧，并分析主要车道
        同时生成可视化视频
        :param output_dir: 输出目录
        """
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # 分析轨迹
        trajectory_lanes = self.analyze_trajectories()
        
        # 准备输出文件
        output_file = Path(output_dir) / f"{self.cam_id}_trajectory_analysis.txt"
        
        def get_main_lane(lane_sequence):
            """统计车道出现频率，返回出现频率最高的车道"""
            lane_count = {}
            for lanes in lane_sequence:
                for lane in lanes:
                    lane_count[lane] = lane_count.get(lane, 0) + 1
            
            if not lane_count:
                return None
                
            return max(lane_count.items(), key=lambda x: x[1])[0]
        
        with open(output_file, 'w', encoding='utf-8') as f:
            for track_id, frame_data in trajectory_lanes.items():
                f.write(f"\n轨迹ID: {track_id}\n")
                f.write("-" * 50 + "\n")
                
                # 统计该轨迹中出现的所有车道
                all_lanes = set()
                for lanes in frame_data.values():
                    all_lanes.update(lanes)
                
                # 获取整个轨迹的主要车道
                all_lane_sequence = list(frame_data.values())
                main_lane = get_main_lane(all_lane_sequence)
                
                f.write(f"经过的所有车道: {sorted(list(all_lanes))}\n")
                f.write(f"主要车道: {main_lane}\n\n")
                
                # 准备视频写入器
                output_path = Path(output_dir) / f"{self.cam_id}_track_{track_id}.mp4"
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                fps = self.cap.get(cv2.CAP_PROP_FPS)
                frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                out = cv2.VideoWriter(str(output_path), fourcc, fps, (frame_width, frame_height))
                
                # 获取轨迹��帧范围
                frame_ids = sorted(frame_data.keys())
                if not frame_ids:
                    continue
                    
                start_frame = frame_ids[0]
                end_frame = frame_ids[-1]
                
                # 设置视频文件的读取位置
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
                
                # 合并连续且车道相同的帧
                current_lanes = frame_data[frame_ids[0]]
                segment_start_frame = start_frame
                prev_frame = start_frame
                current_sequence = []
                
                # 读取并处理每一帧
                for frame_idx in range(start_frame, end_frame + 1):
                    ret, frame = self.cap.read()
                    if not ret:
                        break
                    
                    if frame_idx in frame_data:
                        # 获取当前帧的bbox和车道信息
                        row = self.gt_data[
                            (self.gt_data['Id'] == track_id) & 
                            (self.gt_data['FrameId'] == frame_idx)
                        ].iloc[0]
                        
                        x1 = int(row['X'])
                        y1 = int(row['Y'])
                        x2 = int(x1 + row['Width'])
                        y2 = int(y1 + row['Height'])
                        
                        lanes = frame_data[frame_idx]
                        current_sequence.append(current_lanes)
                        
                        # 如果车道不同或者帧不连续，输出之前的结果并开始新的统计
                        if lanes != current_lanes or frame_idx != prev_frame + 1:
                            segment_main_lane = get_main_lane(current_sequence)
                            if segment_start_frame == prev_frame:
                                f.write(f"Frame {segment_start_frame}: {current_lanes} (主要车道: {segment_main_lane})\n")
                            else:
                                f.write(f"Frame {segment_start_frame}-{prev_frame}: {current_lanes} (主要车道: {segment_main_lane})\n")
                            segment_start_frame = frame_idx
                            current_lanes = lanes
                            current_sequence = []
                        
                        # 绘制bbox和信息
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(
                            frame,
                            f"Track {track_id}, Lanes: {lanes}, Main: {main_lane}",
                            (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            (0, 255, 0),
                            2
                        )
                        
                        prev_frame = frame_idx
                    
                    out.write(frame)
                
                # 处理最后一段
                current_sequence.append(current_lanes)
                segment_main_lane = get_main_lane(current_sequence)
                if segment_start_frame == prev_frame:
                    f.write(f"Frame {segment_start_frame}: {current_lanes} (主要车道: {segment_main_lane})\n")
                else:
                    f.write(f"Frame {segment_start_frame}-{prev_frame}: {current_lanes} (主要车道: {segment_main_lane})\n")
                
                out.release()
                f.write("\n")
                
                # 重置视频读取位置
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    def visualize_uncertain_trajectories(self, output_dir: str = "uncertain_trajectories"):
        """
        为没有明确主要车道的轨迹生成可视化视频
        :param output_dir: 输出目录
        """
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # 分析轨迹
        trajectory_lanes = self.analyze_trajectories()
        
        def get_main_lane(lane_sequence):
            """统计车道出现频率，返回出现频率最高的车道"""
            lane_count = {}
            for lanes in lane_sequence:
                for lane in lanes:
                    lane_count[lane] = lane_count.get(lane, 0) + 1
            
            if not lane_count:
                return None
                
            # 找出出现次数最多的车道
            max_count = max(lane_count.values())
            total_frames = len(lane_sequence)
            
            # 如果最高频率的车道出现次数不足50%，认为没有主要车道
            if max_count / total_frames < 0.5:
                return None
                
            return max(lane_count.items(), key=lambda x: x[1])[0]
        
        # 找出没有主要车道的轨迹
        uncertain_trajectories = {}
        for track_id, frame_data in trajectory_lanes.items():
            all_lane_sequence = list(frame_data.values())
            main_lane = get_main_lane(all_lane_sequence)
            
            if main_lane is None:
                uncertain_trajectories[track_id] = frame_data
        
        if not uncertain_trajectories:
            print("没有发现不确定的轨迹")
            return
        
        # 为每个不确定的轨迹生成视频
        for track_id, frame_data in uncertain_trajectories.items():
            # 获取轨迹的帧范围
            frame_ids = sorted(frame_data.keys())
            start_frame = frame_ids[0]
            end_frame = frame_ids[-1]
            
            # 准备视频写入器
            output_path = Path(output_dir) / f"{self.cam_id}_track_{track_id}.mp4"
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            fps = self.cap.get(cv2.CAP_PROP_FPS)
            frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            out = cv2.VideoWriter(str(output_path), fourcc, fps, (frame_width, frame_height))
            
            # 设置视频文件的读取位置
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            
            # 读取并处理每一帧
            for frame_idx in range(start_frame, end_frame + 1):
                ret, frame = self.cap.read()
                if not ret:
                    break
                    
                if frame_idx in frame_data:
                    # 获取当前帧的bbox
                    row = self.gt_data[
                        (self.gt_data['Id'] == track_id) & 
                        (self.gt_data['FrameId'] == frame_idx)
                    ].iloc[0]
                    
                    x1 = int(row['X'])
                    y1 = int(row['Y'])
                    x2 = int(x1 + row['Width'])
                    y2 = int(y1 + row['Height'])
                    
                    # 绘制bbox和车道信息
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    lanes = frame_data[frame_idx]
                    cv2.putText(
                        frame,
                        f"Track {track_id}, Lanes: {lanes}",
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 0),
                        2
                    )
                
                out.write(frame)
            
            out.release()
            
            # 生成对应的文本说明
            text_path = output_path.with_suffix('.txt')
            with open(text_path, 'w', encoding='utf-8') as f:
                f.write(f"轨迹ID: {track_id}\n")
                f.write("-" * 50 + "\n")
                f.write(f"帧范围: {start_frame} - {end_frame}\n")
                f.write("车道变化情况:\n")
                
                for frame_id in frame_ids:
                    f.write(f"Frame {frame_id}: {frame_data[frame_id]}\n")
        
        # 重置视频读取位置
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

def get_main_lane(lane_sequence: List[List[str]], threshold: float = 0.5, min_lead_ratio: float = 0.1) -> Optional[str]:
    """
    获取车辆的主要车道
    :param lane_sequence: 车道序列，每个元素是一帧中的车道列表
    :param threshold: 判定为主要车道的最小占比阈值
    :param min_lead_ratio: 相比第二多的车道至少领先的比例（如0.1表示领先10%）
    :return: 主要车道ID或None
    """
    # 统计每个车道出现的次数
    lane_count = {}
    total_frames = len(lane_sequence)
    for lanes in lane_sequence:
        for lane in lanes:
            lane_count[lane] = lane_count.get(lane, 0) + 1
    
    if not lane_count or len(lane_count) < 1:
        return None
    
    # 按出现次数排序
    sorted_lanes = sorted(lane_count.items(), key=lambda x: x[1], reverse=True)
    max_lane, max_count = sorted_lanes[0]
    
    # 计算最多车道的占比
    max_ratio = max_count / total_frames
    
    # 如果只有一个车道，只需要检查占比
    if len(sorted_lanes) == 1:
        return max_lane if max_ratio >= threshold else None
    
    # 获取第二多的车道的占比
    second_count = sorted_lanes[1][1]
    second_ratio = second_count / total_frames
    
    # 判断条件：
    # 1. 最多的车道出现占比要超过阈值（默认50%）
    # 2. 最多的车道要比第二多的车道的占比至少高出min_lead_ratio（默认10%）
    if max_ratio >= threshold and (max_ratio - second_ratio) >= min_lead_ratio:
        return max_lane
    
    return None

def analyze_cross_camera_inconsistencies(gt_path: str, video_paths: Dict[str, str], 
                                       threshold: float = 0.5, min_lead_ratio: float = 0.1):
    """
    分析跨摄像头车道不一致的车辆
    :param gt_path: GT文件路径
    :param video_paths: 摄像头ID到视频路径的映射字典
    :param threshold: 判定为主要车道的最小占比阈值
    :param min_lead_ratio: 相比第二多的车道至少领先的帧数
    """
    # 创建每个摄像头的分析器
    analyzers = {
        cam_id: TrajectoryLaneAnalyzer(video_path, gt_path, cam_id)
        for cam_id, video_path in video_paths.items()
    }
    
    # 存储每个摄像头下每个车辆的主要车道和详细信息
    vehicle_lanes = {}  # {vehicle_id: {cam_id: {'main_lane': lane, 'stats': {lane: count}}}}
    
    for cam_id, analyzer in analyzers.items():
        print(f"分析摄像头 {cam_id}...")
        trajectory_lanes = analyzer.analyze_trajectories()
        
        for track_id, frame_data in trajectory_lanes.items():
            # 获取轨迹的起止帧
            frame_ids = sorted(frame_data.keys())
            start_frame = frame_ids[0]
            end_frame = frame_ids[-1]
            
            # 获取该轨迹的所有车道序列
            all_lane_sequence = list(frame_data.values())
            
            # 统计车道信息
            lane_count = {}
            total_frames = len(all_lane_sequence)
            for lanes in all_lane_sequence:
                for lane in lanes:
                    lane_count[lane] = lane_count.get(lane, 0) + 1
            
            # 使用新的函数判断主要车道
            main_lane = get_main_lane(all_lane_sequence, threshold, min_lead_ratio)
            
            if track_id not in vehicle_lanes:
                vehicle_lanes[track_id] = {}
            
            # 保存详细统计信息，包括起止帧
            vehicle_lanes[track_id][cam_id] = {
                'main_lane': main_lane,
                'stats': lane_count,
                'total_frames': total_frames,
                'start_frame': start_frame,
                'end_frame': end_frame
            }
    
    # 找出在多个摄像头中出现且主要车道不一致的车辆
    inconsistent_vehicles = []
    for vehicle_id, cam_lanes in vehicle_lanes.items():
        # 获取所有有主要车道的摄像头
        valid_cams = {
            cam_id: data for cam_id, data in cam_lanes.items() 
            if data['main_lane'] is not None
        }
        
        if len(valid_cams) > 1:  # 在多个摄像头中都有主要车道
            main_lanes = set(data['main_lane'] for data in valid_cams.values())
            if len(main_lanes) > 1:  # 主要车道不一致
                inconsistent_vehicles.append({
                    'vehicle_id': vehicle_id,
                    'camera_data': valid_cams
                })
    
    # 输出结果到文件
    output_path = Path('inconsistent_vehicles_detailed.txt')
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("跨摄像头车道不一致的车辆详细分析：\n")
        f.write("=" * 80 + "\n\n")
        
        for vehicle in inconsistent_vehicles:
            vehicle_id = vehicle['vehicle_id']
            f.write(f"车辆ID: {vehicle_id}\n")
            f.write("-" * 50 + "\n")
            
            # 先输出该车辆在各个摄像头中的主要车道概览
            f.write("主要车道概览:\n")
            for cam_id, data in vehicle['camera_data'].items():
                f.write(f"{cam_id}: {data['main_lane']}\n")
            f.write("\n")
            
            # 然后输出每个摄像头的详细信息
            for cam_id, data in vehicle['camera_data'].items():
                f.write(f"\n摄像头 {cam_id}:\n")
                f.write(f"帧范围: {data['start_frame']} - {data['end_frame']} (共{data['total_frames']}帧)\n")
                f.write(f"主要车道: {data['main_lane']}\n")
                f.write("车道统计:\n")
                total_frames = data['total_frames']
                
                # 按出现次数排序显示车道统计
                for lane, count in sorted(data['stats'].items(), key=lambda x: x[1], reverse=True):
                    ratio = count / total_frames * 100
                    f.write(f"- 车道 {lane}: {count}帧 ({ratio:.1f}%)\n")
            
            f.write("\n" + "=" * 80 + "\n\n")
    
    print(f"\n分析完成！找到 {len(inconsistent_vehicles)} 个车道不一致的车辆")
    print(f"详细结果已保存到: {output_path}")
    
    return inconsistent_vehicles

def visualize_inconsistent_vehicles(gt_path: str, video_paths: Dict[str, str], output_dir: str = "inconsistent_vehicles_vis"):
    """
    可视化跨摄像头车道不一致的车辆，生成2x2布局的视频
    :param gt_path: GT文件路径
    :param video_paths: 摄像头视频路径字典
    :param output_dir: 输出目录
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # 获取不一致的车辆
    print("分析车道不一致的车辆...")
    inconsistent_vehicles = analyze_cross_camera_inconsistencies(gt_path, video_paths)
    print(f"找到 {len(inconsistent_vehicles)} 个车道不一致的车辆")
    
    # 创建视频捕获器
    caps = {
        cam_id: cv2.VideoCapture(path)
        for cam_id, path in video_paths.items()
    }
    
    # 获取视频参数
    frame_width = int(caps['c006'].get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(caps['c006'].get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = caps['c006'].get(cv2.CAP_PROP_FPS)
    
    # 创建2x2布局的画布
    canvas_width = frame_width * 2
    canvas_height = frame_height * 2
    
    # 为每个不一致的车辆创建视频
    for vehicle_idx, vehicle in enumerate(inconsistent_vehicles, 1):
        vehicle_id = vehicle['vehicle_id']
        camera_lanes = vehicle['camera_lanes']
        
        print(f"\n处理车辆 {vehicle_id} ({vehicle_idx}/{len(inconsistent_vehicles)})")
        
        # 创建输出视频写入器
        output_path = Path(output_dir) / f"vehicle_{vehicle_id}_multi_view.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(output_path), fourcc, fps, (canvas_width, canvas_height))
        
        # 获取该车辆在每个摄像头中的帧范围
        print("分析帧范围...")
        frame_ranges = {}
        for cam_id in video_paths.keys():
            gt_data = pd.read_csv(
                gt_path,
                sep='\s+|\t+|,',
                header=None,
                names=['CameraId', 'Id', 'FrameId', 'X', 'Y', 'Width', 'Height', 'Xworld', 'Yworld'],
                engine='python'
            )
            
            # 筛选当前摄像头和车辆的数据
            cam_data = gt_data[
                (gt_data['CameraId'] == int(cam_id[1:])) & 
                (gt_data['Id'] == vehicle_id)
            ]
            
            if not cam_data.empty:
                frame_ranges[cam_id] = {
                    'start': int(cam_data['FrameId'].min()),
                    'end': int(cam_data['FrameId'].max())
                }
        
        # 找到所有摄像头中最早的开始帧和最晚的结束帧
        if frame_ranges:
            global_start = min(r['start'] for r in frame_ranges.values())
            global_end = max(r['end'] for r in frame_ranges.values())
            
            # 设置所有视频的起始位置
            for cap in caps.values():
                cap.set(cv2.CAP_PROP_POS_FRAMES, global_start)
            
            # 处理每一帧
            total_frames = global_end - global_start + 1
            with tqdm(total=total_frames, desc="生成视频") as pbar:
                for frame_idx in range(global_start, global_end + 1):
                    canvas = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)
                    
                    # 读取并处理每个摄像头的画面
                    for i, (cam_id, cap) in enumerate(caps.items()):
                        ret, frame = cap.read()
                        if not ret:
                            continue
                        
                        # 确定当前摄像头画面在画布中的位置
                        row = i // 2
                        col = i % 2
                        y_offset = row * frame_height
                        x_offset = col * frame_width
                        
                        # 如果当前帧在该摄像头的跟踪范围内
                        if cam_id in frame_ranges and frame_ranges[cam_id]['start'] <= frame_idx <= frame_ranges[cam_id]['end']:
                            # 获取bbox数据
                            frame_data = pd.read_csv(
                                gt_path,
                                sep='\s+|\t+|,',
                                header=None,
                                names=['CameraId', 'Id', 'FrameId', 'X', 'Y', 'Width', 'Height', 'Xworld', 'Yworld'],
                                engine='python'
                            )
                            
                            bbox_data = frame_data[
                                (frame_data['CameraId'] == int(cam_id[1:])) &
                                (frame_data['Id'] == vehicle_id) &
                                (frame_data['FrameId'] == frame_idx)
                            ]
                            
                            if not bbox_data.empty:
                                row = bbox_data.iloc[0]
                                x1 = int(row['X'])
                                y1 = int(row['Y'])
                                x2 = int(x1 + row['Width'])
                                y2 = int(y1 + row['Height'])
                                
                                # 绘制bbox和信息
                                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                                lane_info = f"Lane: {camera_lanes.get(cam_id, 'N/A')}"
                                cv2.putText(
                                    frame,
                                    lane_info,
                                    (x1, y1 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.5,
                                    (0, 255, 0),
                                    2
                                )
                        
                        # 添加摄像头标识和进度信息
                        cv2.putText(
                            frame,
                            f"{cam_id} | Frame: {frame_idx}",
                            (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1,
                            (255, 255, 255),
                            2
                        )
                        
                        # 将处理后的帧放入画布
                        canvas[y_offset:y_offset + frame_height, x_offset:x_offset + frame_width] = frame
                    
                    # 写入合并后的帧
                    out.write(canvas)
                    pbar.update(1)
            
            # 重置所有视频的读取位置
            for cap in caps.values():
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            
            print(f"视频已保存到: {output_path}")
        
        out.release()
    
    # 释放所有视频捕获器
    for cap in caps.values():
        cap.release()
    
    print("\n所有视频处理完成！")

def analyze_all_vehicles(gt_path: str, video_paths: Dict[str, str], 
                        threshold: float = 0.5, min_lead_ratio: float = 0.1):
    """
    分析所有摄像头中所有车辆的车道情况
    :param gt_path: GT文件路径
    :param video_paths: 摄像头视频路径字典
    :param threshold: 判定为主要车道的最小占比阈值
    :param min_lead_ratio: 相比第二多的车道至少领先的帧数
    """
    # 创建输出目录
    output_dir = Path("车道分析结果")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 创建每个摄像头的分析器
    analyzers = {
        cam_id: TrajectoryLaneAnalyzer(video_path, gt_path, cam_id)
        for cam_id, video_path in video_paths.items()
    }
    
    # 存储所有车辆在各个摄像头中的信息
    all_vehicles = {}  # {vehicle_id: {cam_id: {'main_lane': lane, 'stats': {lane: count}}}}
    
    # 分析每个摄像头
    for cam_id, analyzer in analyzers.items():
        print(f"分析摄像头 {cam_id}...")
        trajectory_lanes = analyzer.analyze_trajectories()
        
        for track_id, frame_data in trajectory_lanes.items():
            # 获取轨迹的起止帧
            frame_ids = sorted(frame_data.keys())
            start_frame = frame_ids[0]
            end_frame = frame_ids[-1]
            
            # 获取该轨迹的所有车道序列
            all_lane_sequence = list(frame_data.values())
            
            # 统计车道信息
            lane_count = {}
            total_frames = len(all_lane_sequence)
            for lanes in all_lane_sequence:
                for lane in lanes:
                    lane_count[lane] = lane_count.get(lane, 0) + 1
            
            # 使用新的函数判断主要车道
            main_lane = get_main_lane(all_lane_sequence, threshold, min_lead_ratio)
            
            if track_id not in all_vehicles:
                all_vehicles[track_id] = {}
            
            # 保存详细统计信息
            all_vehicles[track_id][cam_id] = {
                'main_lane': main_lane,
                'stats': lane_count,
                'total_frames': total_frames,
                'start_frame': start_frame,
                'end_frame': end_frame
            }
    
    # 输出所有车辆的分析结果
    output_path = output_dir / "所有车辆分析.txt"
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("所有车辆的车道分析结果：\n")
        f.write("=" * 80 + "\n\n")
        
        # 按车辆ID排序
        for vehicle_id in sorted(all_vehicles.keys()):
            f.write(f"车辆ID: {vehicle_id}\n")
            f.write("-" * 50 + "\n")
            
            # 先输出该车辆在各个摄像头中的主要车道概览
            f.write("主要车道概览:\n")
            for cam_id in sorted(video_paths.keys()):  # 按摄像头ID顺序输出
                if cam_id in all_vehicles[vehicle_id]:
                    main_lane = all_vehicles[vehicle_id][cam_id]['main_lane']
                    f.write(f"{cam_id}: {main_lane if main_lane is not None else '无主要车道'}\n")
            f.write("\n")
            
            # 然后输出每个摄像头的详细信息
            for cam_id in sorted(video_paths.keys()):
                if cam_id in all_vehicles[vehicle_id]:
                    data = all_vehicles[vehicle_id][cam_id]
                    f.write(f"\n摄像头 {cam_id}:\n")
                    f.write(f"帧范围: {data['start_frame']} - {data['end_frame']} (共{data['total_frames']}帧)\n")
                    f.write(f"主要车道: {data['main_lane'] if data['main_lane'] is not None else '无主要车道'}\n")
                    f.write("车道统计:\n")
                    
                    # 按出现次数排序显示车道统计
                    total_frames = data['total_frames']
                    for lane, count in sorted(data['stats'].items(), key=lambda x: x[1], reverse=True):
                        ratio = count / total_frames * 100
                        f.write(f"- 车道 {lane}: {count}帧 ({ratio:.1f}%)\n")
            
            f.write("\n" + "=" * 80 + "\n\n")
    
    # 找出无主要车道和车道冲突的车辆
    no_main_lane_vehicles = []  # 在任一摄像头中没有主要车道的车辆
    inconsistent_vehicles = []  # 在不同摄像头中主要车道不一致的车辆
    
    for vehicle_id, cam_data in all_vehicles.items():
        # 检查是否存在无主要车道的情况
        has_no_main_lane = any(
            data['main_lane'] is None 
            for data in cam_data.values()
        )
        if has_no_main_lane:
            no_main_lane_vehicles.append(vehicle_id)
        
        # 检查是否存在车道冲突
        valid_lanes = [
            data['main_lane'] 
            for data in cam_data.values() 
            if data['main_lane'] is not None
        ]
        if len(set(valid_lanes)) > 1:
            inconsistent_vehicles.append(vehicle_id)
    
    # 输出无主要车道的车辆
    no_main_lane_path = output_dir / "无主要车道车辆.txt"
    with open(no_main_lane_path, 'w', encoding='utf-8') as f:
        f.write("无主要车道的车辆分析结果：\n")
        f.write("=" * 80 + "\n\n")
        
        for vehicle_id in sorted(no_main_lane_vehicles):
            f.write(f"车辆ID: {vehicle_id}\n")
            f.write("-" * 50 + "\n")
            
            for cam_id in sorted(video_paths.keys()):
                if cam_id in all_vehicles[vehicle_id]:
                    data = all_vehicles[vehicle_id][cam_id]
                    f.write(f"\n摄像头 {cam_id}:\n")
                    f.write(f"帧范围: {data['start_frame']} - {data['end_frame']} (共{data['total_frames']}帧)\n")
                    f.write(f"主要车道: {data['main_lane'] if data['main_lane'] is not None else '无主要车道'}\n")
                    f.write("车道统计:\n")
                    total_frames = data['total_frames']
                    for lane, count in sorted(data['stats'].items(), key=lambda x: x[1], reverse=True):
                        ratio = count / total_frames * 100
                        f.write(f"- 车道 {lane}: {count}帧 ({ratio:.1f}%)\n")
            f.write("\n" + "=" * 80 + "\n\n")
    
    # 输出车道冲突的车辆
    inconsistent_path = output_dir / "车道冲突车辆.txt"
    with open(inconsistent_path, 'w', encoding='utf-8') as f:
        f.write("车道冲突的车辆分析结果：\n")
        f.write("=" * 80 + "\n\n")
        
        for vehicle_id in sorted(inconsistent_vehicles):
            f.write(f"车辆ID: {vehicle_id}\n")
            f.write("-" * 50 + "\n")
            
            # 先输出概览
            f.write("主要车道概览:\n")
            for cam_id in sorted(video_paths.keys()):
                if cam_id in all_vehicles[vehicle_id]:
                    main_lane = all_vehicles[vehicle_id][cam_id]['main_lane']
                    f.write(f"{cam_id}: {main_lane if main_lane is not None else '无主要车道'}\n")
            f.write("\n")
            
            # 输出详细信息
            for cam_id in sorted(video_paths.keys()):
                if cam_id in all_vehicles[vehicle_id]:
                    data = all_vehicles[vehicle_id][cam_id]
                    f.write(f"\n摄像头 {cam_id}:\n")
                    f.write(f"帧范围: {data['start_frame']} - {data['end_frame']} (共{data['total_frames']}帧)\n")
                    f.write(f"主要车道: {data['main_lane'] if data['main_lane'] is not None else '无主要车道'}\n")
                    f.write("车道统计:\n")
                    total_frames = data['total_frames']
                    for lane, count in sorted(data['stats'].items(), key=lambda x: x[1], reverse=True):
                        ratio = count / total_frames * 100
                        f.write(f"- 车道 {lane}: {count}帧 ({ratio:.1f}%)\n")
            f.write("\n" + "=" * 80 + "\n\n")
    
    print(f"\n分析完成！")
    print(f"共分析了 {len(all_vehicles)} 个车辆")
    print(f"发现 {len(no_main_lane_vehicles)} 个无主要车道的车辆")
    print(f"发现 {len(inconsistent_vehicles)} 个车道冲突的车辆")
    print(f"详细结果已保存到目录: {output_dir}")
    
    return all_vehicles, no_main_lane_vehicles, inconsistent_vehicles

def main():
    # 定义所有摄像头的视频路径
    video_paths = {
        "c006": "/home/chatmindai/project/zhangkun/Fast_Online_MTMCT/dataset/AIC19/validation/S02/c006/vdo.avi",
        "c007": "/home/chatmindai/project/zhangkun/Fast_Online_MTMCT/dataset/AIC19/validation/S02/c007/vdo.avi",
        "c008": "/home/chatmindai/project/zhangkun/Fast_Online_MTMCT/dataset/AIC19/validation/S02/c008/vdo.avi",
        "c009": "/home/chatmindai/project/zhangkun/Fast_Online_MTMCT/dataset/AIC19/validation/S02/c009/vdo.avi"
    }
    
    gt_path = "/home/chatmindai/project/zhangkun/Fast_Online_MTMCT/2. online_MTMC/outputs/ground_truth_validation.txt"
    
    # 分析所有车辆
    analyze_all_vehicles(
        gt_path, 
        video_paths,
        threshold=0.5,      # 主要车道需要占50%以上
        min_lead_ratio=0.1  # 需要比第二多的车道多10%的占比
    )

if __name__ == "__main__":
    main()