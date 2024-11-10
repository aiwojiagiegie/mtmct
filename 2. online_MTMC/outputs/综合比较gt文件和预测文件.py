import numpy as np
from collections import defaultdict
from prettytable import PrettyTable

from opts import opt


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
            gt_frame_ranges[int(target_car_id)][camera_id][0] = min(gt_frame_ranges[int(target_car_id)][camera_id][0],
                                                                    frame_number)
            gt_frame_ranges[int(target_car_id)][camera_id][1] = max(gt_frame_ranges[int(target_car_id)][camera_id][1],
                                                                    frame_number)

            matching_preds = pred_data[(pred_data[:, 0] == camera_id) & (pred_data[:, 2] == frame_number)]

            for pred_row in matching_preds:
                pred_box = pred_row[3:7]
                if iou(gt_box, pred_box) > 0.5:
                    pred_car_id = int(pred_row[1])
                    matched_pred_ids[int(target_car_id)][pred_car_id][camera_id][0] = min(
                        matched_pred_ids[int(target_car_id)][pred_car_id][camera_id][0], frame_number)
                    matched_pred_ids[int(target_car_id)][pred_car_id][camera_id][1] = max(
                        matched_pred_ids[int(target_car_id)][pred_car_id][camera_id][1], frame_number)
                    break

    return matched_pred_ids, gt_frame_ranges


if __name__ == '__main__':
    version = opt.version
    version = '6'
    gt_file = '/home/chatmindai/project/zhangkun/Fast_Online_MTMCT/2. online_MTMC/outputs/ground_truth_validation.txt'
    pred_file = f'/home/chatmindai/project/zhangkun/Fast_Online_MTMCT/2. online_MTMC/outputs/result/version/v{version}.txt'
    baseline_file = 'result/baseline.txt'

    all_matched_ids, gt_frame_ranges = compare_all_cars(gt_file, pred_file)
    baseline_matched_ids, _ = compare_all_cars(gt_file, baseline_file)

    # 创建表格，添加baseline列
    table = PrettyTable()
    table.field_names = ["   GT车辆ID   ", "   预测ID   ", "   baseline ID   ", "  相机ID  ", "  GT帧范围  ",
                         "  预测帧范围  ", "  基准帧范围  "]

    prev_gt_car_id = None
    prev_baseline_id = None  # 添加这一行来跟踪上一个baseline ID
    for gt_car_id, pred_info in all_matched_ids.items():
        if prev_gt_car_id is not None:
            table.add_row(["--------", "--------", "--------", "--------", "--------", "--------", "--------"])
            prev_baseline_id = None  # 重置prev_baseline_id

        baseline_info = baseline_matched_ids.get(gt_car_id, {})
        pred_car_ids = list(pred_info.keys())
        baseline_car_ids = list(baseline_info.keys())

        if pred_car_ids or baseline_car_ids:
            first_pred = True
            processed_cameras = set()

            # 处理预测结果
            for pred_car_id, camera_info in pred_info.items():
                first_camera = True
                for camera_id, frame_range in sorted(camera_info.items()):
                    processed_cameras.add(camera_id)
                    gt_range = gt_frame_ranges[gt_car_id][camera_id]
                    baseline_range = [-1, -1]
                    baseline_id = ""

                    # 查找对应的baseline结果
                    for b_id, b_camera_info in baseline_info.items():
                        if camera_id in b_camera_info:
                            baseline_range = b_camera_info[camera_id]
                            baseline_id = b_id
                            break

                    # 如果baseline_id与上一个相同，则显示为空字符串
                    display_baseline_id = "" if baseline_id == prev_baseline_id else baseline_id
                    prev_baseline_id = baseline_id

                    if first_pred and first_camera:
                        table.add_row([gt_car_id, pred_car_id, display_baseline_id, camera_id,
                                       f"{int(gt_range[0])}-{int(gt_range[1])}",
                                       f"{int(frame_range[0])}-{int(frame_range[1])}",
                                       f"{int(baseline_range[0])}-{int(baseline_range[1])}" if baseline_range[
                                                                                                   0] != -1 else "-"])
                        first_pred = False
                    elif first_camera:
                        table.add_row(["", pred_car_id, display_baseline_id, camera_id,
                                       f"{int(gt_range[0])}-{int(gt_range[1])}",
                                       f"{int(frame_range[0])}-{int(frame_range[1])}",
                                       f"{int(baseline_range[0])}-{int(baseline_range[1])}" if baseline_range[
                                                                                                   0] != -1 else "-"])
                    else:
                        table.add_row(["", "", display_baseline_id, camera_id,
                                       f"{int(gt_range[0])}-{int(gt_range[1])}",
                                       f"{int(frame_range[0])}-{int(frame_range[1])}",
                                       f"{int(baseline_range[0])}-{int(baseline_range[1])}" if baseline_range[
                                                                                                   0] != -1 else "-"])
                    first_camera = False

            # 处理仅在baseline中出现的相机
            for b_id, b_camera_info in baseline_info.items():
                for camera_id, baseline_range in sorted(b_camera_info.items()):
                    if camera_id not in processed_cameras:
                        gt_range = gt_frame_ranges[gt_car_id][camera_id]
                        display_baseline_id = "" if b_id == prev_baseline_id else b_id
                        prev_baseline_id = b_id
                        table.add_row(["", "无匹配", display_baseline_id, camera_id,
                                       f"{int(gt_range[0])}-{int(gt_range[1])}",
                                       "-",
                                       f"{int(baseline_range[0])}-{int(baseline_range[1])}"])
        else:
            gt_ranges = gt_frame_ranges[gt_car_id]
            for camera_id, gt_range in sorted(gt_ranges.items()):
                table.add_row([gt_car_id if camera_id == list(gt_ranges.keys())[0] else "",
                               "无匹配", "无匹配", camera_id,
                               f"{int(gt_range[0])}-{int(gt_range[1])}", "-", "-"])

        prev_gt_car_id = gt_car_id

    print(table)

    # 统计信息
    total_gt_cars = len(all_matched_ids)
    matched_gt_cars = sum(1 for pred_ids in all_matched_ids.values() if pred_ids)
    baseline_matched_cars = sum(1 for pred_ids in baseline_matched_ids.values() if pred_ids)
    unmatched_gt_cars = total_gt_cars - matched_gt_cars

    print(f"\n统计信息:")
    print(f"总gt车辆数: {total_gt_cars}")
    print(f"有匹配的gt车辆数: {matched_gt_cars}")
    print(f"baseline匹配的gt车辆数: {baseline_matched_cars}")
    print(f"未匹配的gt车辆数: {unmatched_gt_cars}")
    print(f"匹配率: {matched_gt_cars / total_gt_cars * 100:.2f}%")
    print(f"baseline匹配率: {baseline_matched_cars / total_gt_cars * 100:.2f}%")
