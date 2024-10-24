from prettytable import PrettyTable
from datetime import datetime, timedelta
import cv2
import numpy as np

def get_bbox_data(txt_path):
    bbox_data = {}
    min_area = float('inf')  # 初始化最小面积为无穷大
    
    with open(txt_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 0:
                break
            camera_id = int(parts[0])
            frame_number = int(parts[2])
            car_id = int(parts[1])
            left = int(parts[3])
            top = int(parts[4])
            width = int(parts[5])
            height = int(parts[6])
            
            # 计算并更新最小面积
            area = width * height
            min_area = min(min_area, area)
            
            if camera_id not in bbox_data:
                bbox_data[camera_id] = {}
            if frame_number not in bbox_data[camera_id]:
                bbox_data[camera_id][frame_number] = []
            bbox_data[camera_id][frame_number].append((left, top, width, height, car_id))
    
    return bbox_data, min_area

def calculate_max_frame_diff(bbox_data):
    car_appearances = {}
    max_frame_diff = 0
    max_diff_info = None

    for camera_id in sorted(bbox_data.keys()):
        for frame_number in sorted(bbox_data[camera_id].keys()):
            for _, _, _, _, car_id in bbox_data[camera_id][frame_number]:
                if car_id not in car_appearances:
                    car_appearances[car_id] = {}
                if camera_id not in car_appearances[car_id]:
                    car_appearances[car_id][camera_id] = {'first': frame_number, 'last': frame_number}
                else:
                    car_appearances[car_id][camera_id]['last'] = frame_number

    for car_id, appearances in car_appearances.items():
        camera_ids = sorted(appearances.keys())
        for i in range(len(camera_ids) - 1):
            current_camera = camera_ids[i]
            next_camera = camera_ids[i + 1]
            if next_camera - current_camera == 1:  # 确保是相邻的摄像头
                frame_diff = appearances[next_camera]['first'] - appearances[current_camera]['last']
                if frame_diff > max_frame_diff:
                    max_frame_diff = frame_diff
                    max_diff_info = {
                        'car_id': car_id,
                        'current_camera': current_camera,
                        'next_camera': next_camera,
                        'last_frame': appearances[current_camera]['last'],
                        'first_frame': appearances[next_camera]['first']
                    }

    return max_frame_diff, max_diff_info

def find_overlapping_vehicles(bbox_data):
    car_appearances = {}
    overlapping_vehicles = []

    # 构建车辆出现信息
    for camera_id in sorted(bbox_data.keys()):
        for frame_number in sorted(bbox_data[camera_id].keys()):
            for _, _, _, _, car_id in bbox_data[camera_id][frame_number]:
                if car_id not in car_appearances:
                    car_appearances[car_id] = {}
                if camera_id not in car_appearances[car_id]:
                    car_appearances[car_id][camera_id] = {'first': frame_number, 'last': frame_number}
                else:
                    car_appearances[car_id][camera_id]['last'] = frame_number

    # 查找帧重叠的车辆
    for car_id, appearances in car_appearances.items():
        camera_ids = sorted(appearances.keys())
        for i in range(len(camera_ids) - 1):
            current_camera = camera_ids[i]
            next_camera = camera_ids[i + 1]
            if next_camera - current_camera == 1:  # 确保是相邻的摄像头
                if appearances[current_camera]['last'] >= appearances[next_camera]['first']:
                    frame_diff = appearances[current_camera]['last'] - appearances[next_camera]['first']
                    overlapping_vehicles.append({
                        'car_id': car_id,
                        'current_camera': current_camera,
                        'next_camera': next_camera,
                        'last_frame': appearances[current_camera]['last'],
                        'first_frame': appearances[next_camera]['first'],
                        'frame_diff': frame_diff  # 添加帧差值
                    })

    return overlapping_vehicles

def find_earliest_and_latest_frames(bbox_data):
    earliest_frame = float('inf')
    latest_frame = float('-inf')
    earliest_frame_info = None
    latest_frame_info = None

    for camera_id in bbox_data:
        for frame_number in bbox_data[camera_id]:
            if frame_number < earliest_frame:
                earliest_frame = frame_number
                earliest_frame_info = {
                    'camera_id': camera_id,
                    'frame_number': frame_number,
                    'car_ids': [car[4] for car in bbox_data[camera_id][frame_number]]
                }
            if frame_number > latest_frame:
                latest_frame = frame_number
                latest_frame_info = {
                    'camera_id': camera_id,
                    'frame_number': frame_number,
                    'car_ids': [car[4] for car in bbox_data[camera_id][frame_number]]
                }

    return earliest_frame_info, latest_frame_info

def find_earliest_and_latest_start_frames(bbox_data):
    earliest_start = float('inf')
    latest_start = float('-inf')
    earliest_start_info = None
    latest_start_info = None
    car_first_appearances = {}

    for camera_id in bbox_data:
        for frame_number in sorted(bbox_data[camera_id].keys()):
            for _, _, _, _, car_id in bbox_data[camera_id][frame_number]:
                if car_id not in car_first_appearances:
                    car_first_appearances[car_id] = (camera_id, frame_number)
                    
                    if frame_number < earliest_start:
                        earliest_start = frame_number
                        earliest_start_info = {
                            'camera_id': camera_id,
                            'frame_number': frame_number,
                            'car_id': car_id
                        }
                    
                    if frame_number > latest_start:
                        latest_start = frame_number
                        latest_start_info = {
                            'camera_id': camera_id,
                            'frame_number': frame_number,
                            'car_id': car_id
                        }

    return earliest_start_info, latest_start_info

def find_latest_start_frames(bbox_data, camera_ids, top_n=10):
    car_first_appearances = {}

    for camera_id in camera_ids:
        if camera_id not in bbox_data:
            continue
        for frame_number in sorted(bbox_data[camera_id].keys()):
            for _, _, _, _, car_id in bbox_data[camera_id][frame_number]:
                if car_id not in car_first_appearances:
                    car_first_appearances[car_id] = {
                        'camera_id': camera_id,
                        'frame_number': frame_number,
                        'car_id': car_id
                    }

    # 按起始帧号降序排序
    sorted_appearances = sorted(car_first_appearances.values(), 
                                key=lambda x: x['frame_number'], 
                                reverse=True)

    return sorted_appearances[:top_n]

def find_vehicles_not_in_camera(bbox_data, target_cameras, exclude_camera, top_n=10):
    vehicles_in_target = set()
    vehicles_in_exclude = set()
    car_first_appearances = {}

    # 收集在目标摄像头和排除摄像头中出现的车辆
    for camera_id in bbox_data:
        for frame_number in bbox_data[camera_id]:
            for _, _, _, _, car_id in bbox_data[camera_id][frame_number]:
                if camera_id in target_cameras:
                    vehicles_in_target.add(car_id)
                    if car_id not in car_first_appearances or frame_number < car_first_appearances[car_id]['frame_number']:
                        car_first_appearances[car_id] = {
                            'camera_id': camera_id,
                            'frame_number': frame_number,
                            'car_id': car_id
                        }
                elif camera_id == exclude_camera:
                    vehicles_in_exclude.add(car_id)

    # 找出在目标摄像头中出现但不在排除摄像头中出现的车辆
    vehicles_of_interest = vehicles_in_target - vehicles_in_exclude

    # 筛选出感兴趣的车辆的首次出现信息
    filtered_appearances = [info for car_id, info in car_first_appearances.items() if car_id in vehicles_of_interest]

    # 按起始帧号降序排序
    sorted_appearances = sorted(filtered_appearances, key=lambda x: x['frame_number'], reverse=True)

    return sorted_appearances[:top_n]

def load_lane_images(camera_ids):
    lane_images = {}
    for camera_id in camera_ids:
        image_path = f'/home/chatmindai/project/zhangkun/Fast_Online_MTMCT/2. online_MTMC/preliminary/devied_zones/{camera_id}.png'
        lane_images[camera_id] = cv2.imread(image_path)
    return lane_images

def get_lane(image, x, y):
    color = image[y, x]
    if np.all(color[2] >= 200):  # 红色
        return "右车道"
    elif np.all(color[0] >= 200):  # 蓝色
        return "左车道"
    else:
        return "未知"

def analyze_vehicle_frames(bbox_data):
    lane_images = load_lane_images([f'4{i}' for i in range(1, 7)])  # 加载所有摄像头的车道图像
    vehicle_frame_analysis = []
    lane_change_vehicles = []

    for car_id in set(car_id for camera in bbox_data.values() for frame in camera.values() for _, _, _, _, car_id in frame):
        for camera_id, frames in bbox_data.items():
            car_frames = [frame for frame, boxes in frames.items() if any(cid == car_id for _, _, _, _, cid in boxes)]
            if car_frames:
                start_frame = min(car_frames)
                end_frame = max(car_frames)
                frame_diff = end_frame - start_frame

                # 获取车辆在该摄像头中的第一帧和最后一帧的位置
                start_box = next(box for box in frames[start_frame] if box[4] == car_id)
                end_box = next(box for box in frames[end_frame] if box[4] == car_id)

                # 计算车辆的中心点
                start_center = (int(start_box[0] + start_box[2] / 2), int(start_box[1] + start_box[3] / 2))
                end_center = (int(end_box[0] + end_box[2] / 2), int(end_box[1] + end_box[3] / 2))

                # 获取车道信息
                lane_image = lane_images[f'{camera_id}']
                start_lane = get_lane(lane_image, start_center[0], start_center[1])
                end_lane = get_lane(lane_image, end_center[0], end_center[1])

                # 检查是否发生车道变化
                if start_lane != end_lane and start_lane != "未知" and end_lane != "未知":
                    lane_change_vehicles.append({
                        'car_id': car_id,
                        'camera_id': camera_id,
                        'start_frame': start_frame,
                        'end_frame': end_frame,
                        'start_lane': start_lane,
                        'end_lane': end_lane
                    })

                vehicle_frame_analysis.append({
                    'car_id': car_id,
                    'camera_id': camera_id,
                    'start_frame': start_frame,
                    'end_frame': end_frame,
                    'frame_diff': frame_diff,
                    'start_lane': start_lane,
                    'end_lane': end_lane
                })

    return vehicle_frame_analysis, lane_change_vehicles

def calculate_frame_diff(camera_sync, camera_id, frame_number):
    sync_frame = camera_sync[str(camera_id)]['frame']
    sync_time = camera_sync[str(camera_id)]['time']
    return frame_number - sync_frame

def calculate_max_frame_diff_with_sync(bbox_data, camera_sync):
    car_appearances = {}
    max_frame_diff = 0
    max_diff_info = None

    for camera_id in sorted(bbox_data.keys()):
        for frame_number in sorted(bbox_data[camera_id].keys()):
            for _, _, _, _, car_id in bbox_data[camera_id][frame_number]:
                if car_id not in car_appearances:
                    car_appearances[car_id] = {}
                if camera_id not in car_appearances[car_id]:
                    car_appearances[car_id][camera_id] = {
                        'first': calculate_frame_diff(camera_sync, camera_id, frame_number),
                        'last': calculate_frame_diff(camera_sync, camera_id, frame_number)
                    }
                else:
                    car_appearances[car_id][camera_id]['last'] = calculate_frame_diff(camera_sync, camera_id, frame_number)

    for car_id, appearances in car_appearances.items():
        camera_ids = sorted(appearances.keys())
        for i in range(len(camera_ids) - 1):
            current_camera = camera_ids[i]
            next_camera = camera_ids[i + 1]
            if next_camera - current_camera == 1:  # 确保是相邻的摄像头
                frame_diff = appearances[next_camera]['first'] - appearances[current_camera]['last']
                if frame_diff > max_frame_diff:
                    max_frame_diff = frame_diff
                    max_diff_info = {
                        'car_id': car_id,
                        'current_camera': current_camera,
                        'next_camera': next_camera,
                        'last_frame': appearances[current_camera]['last'],
                        'first_frame': appearances[next_camera]['first']
                    }

    return max_frame_diff, max_diff_info

def find_overlapping_vehicles_with_sync(bbox_data, camera_sync):
    car_appearances = {}
    overlapping_vehicles = []

    for camera_id in sorted(bbox_data.keys()):
        for frame_number in sorted(bbox_data[camera_id].keys()):
            for _, _, _, _, car_id in bbox_data[camera_id][frame_number]:
                if car_id not in car_appearances:
                    car_appearances[car_id] = {}
                if camera_id not in car_appearances[car_id]:
                    car_appearances[car_id][camera_id] = {
                        'first': calculate_frame_diff(camera_sync, camera_id, frame_number),
                        'last': calculate_frame_diff(camera_sync, camera_id, frame_number)
                    }
                else:
                    car_appearances[car_id][camera_id]['last'] = calculate_frame_diff(camera_sync, camera_id, frame_number)

    for car_id, appearances in car_appearances.items():
        camera_ids = sorted(appearances.keys())
        for i in range(len(camera_ids) - 1):
            current_camera = camera_ids[i]
            next_camera = camera_ids[i + 1]
            if next_camera - current_camera == 1:  # 确保是相邻的摄像头
                if appearances[current_camera]['last'] >= appearances[next_camera]['first']:
                    frame_diff = appearances[current_camera]['last'] - appearances[next_camera]['first']
                    overlapping_vehicles.append({
                        'car_id': car_id,
                        'current_camera': current_camera,
                        'next_camera': next_camera,
                        'last_frame': appearances[current_camera]['last'],
                        'first_frame': appearances[next_camera]['first'],
                        'frame_diff': frame_diff
                    })

    return overlapping_vehicles

if __name__ == '__main__':
    txt_path = './test_gt.txt'
    bbox_data, min_area = get_bbox_data(txt_path)
    print(f"最小的边界框面积是: {min_area}")

    max_frame_diff, max_diff_info = calculate_max_frame_diff(bbox_data)
    print(f"相邻摄像头之间的最大帧ID差是: {max_frame_diff}")
    print("最大帧ID差详细信息:")
    max_diff_table = PrettyTable()
    max_diff_table.field_names = ["车辆ID", "相邻摄像头", "摄像头A最后一帧", "摄像头B第一帧"]
    max_diff_table.add_row([
        max_diff_info['car_id'],
        f"{max_diff_info['current_camera']} 和 {max_diff_info['next_camera']}",
        max_diff_info['last_frame'],
        max_diff_info['first_frame']
    ])
    print(max_diff_table)

    overlapping_vehicles = find_overlapping_vehicles(bbox_data)
    print("\n在相邻摄像头中出现帧重叠的车辆:")
    overlap_table = PrettyTable()
    overlap_table.field_names = ["车辆ID", "相邻摄像头", "摄像头A最后一帧", "摄像头B第一帧", "帧差值"]
    for vehicle in overlapping_vehicles:
        overlap_table.add_row([
            vehicle['car_id'],
            f"{vehicle['current_camera']} 和 {vehicle['next_camera']}",
            vehicle['last_frame'],
            vehicle['first_frame'],
            vehicle['frame_diff']
        ])
    print(overlap_table)

    earliest_frame_info, latest_frame_info = find_earliest_and_latest_frames(bbox_data)
    
    print("\n所有轨迹ID中起始帧最早的信息:")
    earliest_frame_table = PrettyTable()
    earliest_frame_table.field_names = ["摄像头ID", "帧号", "车辆ID"]
    earliest_frame_table.add_row([
        earliest_frame_info['camera_id'],
        earliest_frame_info['frame_number'],
        ', '.join(map(str, earliest_frame_info['car_ids']))
    ])
    print(earliest_frame_table)

    print("\n所有轨迹ID中结束帧最晚的信息:")
    latest_frame_table = PrettyTable()
    latest_frame_table.field_names = ["摄像头ID", "帧号", "车辆ID"]
    latest_frame_table.add_row([
        latest_frame_info['camera_id'],
        latest_frame_info['frame_number'],
        ', '.join(map(str, latest_frame_info['car_ids']))
    ])
    print(latest_frame_table)

    earliest_start_info, latest_start_info = find_earliest_and_latest_start_frames(bbox_data)
    
    print("\n所有轨迹ID中起始帧最早的信息:")
    earliest_start_table = PrettyTable()
    earliest_start_table.field_names = ["摄像头ID", "帧号", "车辆ID"]
    earliest_start_table.add_row([
        earliest_start_info['camera_id'],
        earliest_start_info['frame_number'],
        earliest_start_info['car_id']
    ])
    print(earliest_start_table)

    print("\n所有轨迹ID中起始帧最晚的信息:")
    latest_start_table = PrettyTable()
    latest_start_table.field_names = ["摄像头ID", "帧号", "车辆ID"]
    latest_start_table.add_row([
        latest_start_info['camera_id'],
        latest_start_info['frame_number'],
        latest_start_info['car_id']
    ])
    print(latest_start_table)

    # 指定要查找的摄像头ID列表
    target_camera_ids = [42, 43]  # 这里可以根据需要修改摄像头ID

    # 获取并打印指定摄像头中起始帧最晚的前10个轨迹
    latest_start_frames = find_latest_start_frames(bbox_data, target_camera_ids, top_n=10)
    
    print(f"\n摄像头 {', '.join(map(str, target_camera_ids))} 中起始帧最晚的前10个轨迹信息（降序排列）:")
    latest_start_table = PrettyTable()
    latest_start_table.field_names = ["排名", "摄像头ID", "帧号", "车辆ID"]
    for i, info in enumerate(latest_start_frames, 1):
        latest_start_table.add_row([
            i,
            info['camera_id'],
            info['frame_number'],
            info['car_id']
        ])
    print(latest_start_table)

    # 指定目标摄像头和排除摄像头
    target_cameras = [42, 43, 44, 45, 46]
    exclude_camera = 41

    # 获取并打印符合条件的车辆��起始帧最晚的前10个
    latest_start_frames = find_vehicles_not_in_camera(bbox_data, target_cameras, exclude_camera, top_n=10)
    
    print(f"\n在摄像头 {', '.join(map(str, target_cameras))} 中出现但未在摄像头 {exclude_camera} 中出现的车辆中，起始帧最晚的前10个:")
    latest_start_table = PrettyTable()
    latest_start_table.field_names = ["排名", "摄像头ID", "帧号", "车辆ID"]
    for i, info in enumerate(latest_start_frames, 1):
        latest_start_table.add_row([
            i,
            info['camera_id'],
            info['frame_number'],
            info['car_id']
        ])
    print(latest_start_table)

    # 分析每个车辆在每个摄像头中的帧情况和车道变化
    vehicle_frame_analysis, lane_change_vehicles = analyze_vehicle_frames(bbox_data)
    
    print("\n每个车辆在每个摄像头中的起止帧情况、帧差值及车道信息:")
    vehicle_frame_table = PrettyTable()
    vehicle_frame_table.field_names = ["车辆ID", "摄像头ID", "起始帧", "结束帧", "帧差值", "起始车道", "结束车道"]
    
    sorted_analysis = sorted(vehicle_frame_analysis, key=lambda x: (x['car_id'], x['camera_id']))
    current_car_id = None
    
    for info in sorted_analysis:
        if current_car_id is not None and current_car_id != info['car_id']:
            # 在不同车辆ID之间添加分隔行
            vehicle_frame_table.add_row(["--------"] * 7)
        
        vehicle_frame_table.add_row([
            info['car_id'],
            info['camera_id'],
            info['start_frame'],
            info['end_frame'],
            info['frame_diff'],
            info['start_lane'],
            info['end_lane']
        ])
        
        current_car_id = info['car_id']
    
    print(vehicle_frame_table)

    # 打印车道发生变化的车辆信息
    print("\n车道发生变化的车辆信息:")
    lane_change_table = PrettyTable()
    lane_change_table.field_names = ["车辆ID", "摄像头ID", "起始帧", "结束帧", "起始车道", "结束车道"]
    
    for info in lane_change_vehicles:
        lane_change_table.add_row([
            info['car_id'],
            info['camera_id'],
            info['start_frame'],
            info['end_frame'],
            info['start_lane'],
            info['end_lane']
        ])
    
    print(lane_change_table)

    # 新增的使用同步信息的函数调用
    camera_sync = {
        '41': {'frame': 109, 'time': datetime.strptime('15:30:00', '%H:%M:%S')},
        '42': {'frame': 79, 'time': datetime.strptime('15:30:00', '%H:%M:%S')},
        '43': {'frame': 77, 'time': datetime.strptime('15:30:00', '%H:%M:%S')},
        '44': {'frame': 207, 'time': datetime.strptime('15:30:00', '%H:%M:%S')},
        '45': {'frame': 119, 'time': datetime.strptime('15:30:00', '%H:%M:%S')},
        '46': {'frame': 118, 'time': datetime.strptime('15:30:00', '%H:%M:%S')}
    }

    max_frame_diff_sync, max_diff_info_sync = calculate_max_frame_diff_with_sync(bbox_data, camera_sync)
    print(f"\n使用同步信息计算的相邻摄像头之间的最大帧差是: {max_frame_diff_sync}")
    print("最大帧差详细信息:")
    max_diff_sync_table = PrettyTable()
    max_diff_sync_table.field_names = ["车辆ID", "相邻摄像头", "摄像头A最后一帧", "摄像头B第一帧"]
    max_diff_sync_table.add_row([
        max_diff_info_sync['car_id'],
        f"{max_diff_info_sync['current_camera']} 和 {max_diff_info_sync['next_camera']}",
        max_diff_info_sync['last_frame'],
        max_diff_info_sync['first_frame']
    ])
    print(max_diff_sync_table)

    overlapping_vehicles_sync = find_overlapping_vehicles_with_sync(bbox_data, camera_sync)
    print("\n使用同步信息计算的在相邻摄像头中出现帧重叠的车辆:")
    overlap_sync_table = PrettyTable()
    overlap_sync_table.field_names = ["车辆ID", "相邻摄像头", "摄像头A最后一帧", "摄像头B第一帧", "帧差值"]
    for vehicle in overlapping_vehicles_sync:
        overlap_sync_table.add_row([
            vehicle['car_id'],
            f"{vehicle['current_camera']} 和 {vehicle['next_camera']}",
            vehicle['last_frame'],
            vehicle['first_frame'],
            vehicle['frame_diff']
        ])
    print(overlap_sync_table)

