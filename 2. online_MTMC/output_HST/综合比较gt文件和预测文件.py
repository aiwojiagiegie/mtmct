import numpy as np
from collections import defaultdict
from prettytable import PrettyTable

def iou(box1, box2):
    """计算两个边界框的IOU"""
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    
    xi1 = max(x1, x2)
    yi1 = max(y1, y2)
    xi2 = min(x1 + w1, x2 + w2)
    yi2 = min(y1 + h1, y2 + h2)
    
    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    box1_area = w1 * h1
    box2_area = w2 * h2
    
    iou = inter_area / (box1_area + box2_area - inter_area)
    return iou

def compare_all_cars(gt_file, pred_file):
    gt_data = np.loadtxt(gt_file, delimiter=' ')
    pred_data = np.loadtxt(pred_file, delimiter=' ')
    
    all_gt_car_ids = np.unique(gt_data[:, 1])
    matched_pred_ids = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: [float('inf'), -float('inf')])))
    gt_frame_ranges = defaultdict(lambda: defaultdict(lambda: [float('inf'), -float('inf')]))
    
    for target_car_id in all_gt_car_ids:
        target_gt = gt_data[gt_data[:, 1] == target_car_id]
        
        for gt_row in target_gt:
            camera_id, _, frame_number = gt_row[:3]
            camera_id = int(camera_id)
            frame_number = int(frame_number)
            gt_box = gt_row[3:7]
            
            # 更新GT帧范围
            gt_frame_ranges[int(target_car_id)][camera_id][0] = min(gt_frame_ranges[int(target_car_id)][camera_id][0], frame_number)
            gt_frame_ranges[int(target_car_id)][camera_id][1] = max(gt_frame_ranges[int(target_car_id)][camera_id][1], frame_number)
            
            matching_preds = pred_data[(pred_data[:, 0] == camera_id) & (pred_data[:, 2] == frame_number)]
            
            for pred_row in matching_preds:
                pred_box = pred_row[3:7]
                if iou(gt_box, pred_box) > 0.5:
                    pred_car_id = int(pred_row[1])
                    matched_pred_ids[int(target_car_id)][pred_car_id][camera_id][0] = min(matched_pred_ids[int(target_car_id)][pred_car_id][camera_id][0], frame_number)
                    matched_pred_ids[int(target_car_id)][pred_car_id][camera_id][1] = max(matched_pred_ids[int(target_car_id)][pred_car_id][camera_id][1], frame_number)
                    break
    
    return matched_pred_ids, gt_frame_ranges

# 使用示例
gt_file = '/home/chatmindai/project/zhangkun/Fast_Online_MTMCT/2. online_MTMC/output_HST/test_gt.txt'
pred_file = '/home/chatmindai/project/zhangkun/Fast_Online_MTMCT/2. online_MTMC/output_HST/result/version/vn17.txt'

all_matched_ids, gt_frame_ranges = compare_all_cars(gt_file, pred_file)

# 创建表格
table = PrettyTable()
table.field_names = ["GT Car ID", "Pred Car ID", "Camera ID", "GT Frame Range", "Pred Frame Range"]

prev_gt_car_id = None
for gt_car_id, pred_info in all_matched_ids.items():
    if prev_gt_car_id is not None:
        table.add_row(["--------", "--------", "--------", "--------", "--------"])
    
    pred_car_ids = list(pred_info.keys())
    if pred_car_ids:
        first_pred = True
        for pred_car_id, camera_info in pred_info.items():
            first_camera = True
            for camera_id, frame_range in sorted(camera_info.items()):
                gt_range = gt_frame_ranges[gt_car_id][camera_id]
                if first_pred and first_camera:
                    table.add_row([gt_car_id, pred_car_id, camera_id, f"{int(gt_range[0])}-{int(gt_range[1])}", f"{int(frame_range[0])}-{int(frame_range[1])}"])
                    first_pred = False
                elif first_camera:
                    table.add_row(["", pred_car_id, camera_id, f"{int(gt_range[0])}-{int(gt_range[1])}", f"{int(frame_range[0])}-{int(frame_range[1])}"])
                else:
                    table.add_row(["", "", camera_id, f"{int(gt_range[0])}-{int(gt_range[1])}", f"{int(frame_range[0])}-{int(frame_range[1])}"])
                first_camera = False
    else:
        gt_ranges = gt_frame_ranges[gt_car_id]
        for camera_id, gt_range in sorted(gt_ranges.items()):
            table.add_row([gt_car_id if camera_id == list(gt_ranges.keys())[0] else "", "无匹配", camera_id, f"{int(gt_range[0])}-{int(gt_range[1])}", "-"])
    
    prev_gt_car_id = gt_car_id

print(table)

# 统计信息
total_gt_cars = len(all_matched_ids)
matched_gt_cars = sum(1 for pred_ids in all_matched_ids.values() if pred_ids)
unmatched_gt_cars = total_gt_cars - matched_gt_cars

print(f"\n统计信息:")
print(f"总gt车辆数: {total_gt_cars}")
print(f"有匹配的gt车辆数: {matched_gt_cars}")
print(f"未匹配的gt车辆数: {unmatched_gt_cars}")
print(f"匹配率: {matched_gt_cars / total_gt_cars * 100:.2f}%")
