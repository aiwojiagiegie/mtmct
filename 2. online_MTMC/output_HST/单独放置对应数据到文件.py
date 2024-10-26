import numpy as np
import os

def extract_and_save_data(gt_file, pred_file, gt_car_id, pred_car_id, output_dir):
    # 读取文件
    gt_data = np.loadtxt(gt_file, delimiter=' ')
    pred_data = np.loadtxt(pred_file, delimiter=' ')

    # 提取指定car_id的数据
    gt_car_data = gt_data[gt_data[:, 1] == gt_car_id]
    pred_car_data = pred_data[pred_data[:, 1] == pred_car_id]

    # 排序：先按摄像头ID（第一列），然后按帧ID（第三列）
    gt_car_data = gt_car_data[np.lexsort((gt_car_data[:, 2], gt_car_data[:, 0]))]
    pred_car_data = pred_car_data[np.lexsort((pred_car_data[:, 2], pred_car_data[:, 0]))]

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 保存提取的数据到新文件
    np.savetxt(os.path.join(output_dir, "gt.txt"), gt_car_data, fmt='%d', delimiter=' ')
    np.savetxt(os.path.join(output_dir, "pred.txt"), pred_car_data, fmt='%d', delimiter=' ')

    print(f"已将gt car_id {gt_car_id}和pred car_id {pred_car_id}的数据保存到 {output_dir} 目录下的gt.txt和pred.txt文件中。")
    print("数据已按摄像头ID和帧ID排序。")

# 使用示例
gt_file = '/home/chatmindai/project/zhangkun/Fast_Online_MTMCT/2. online_MTMC/output_HST/test_gt.txt'
pred_file = '/home/chatmindai/project/zhangkun/Fast_Online_MTMCT/2. online_MTMC/output_HST/result/version/vn17.txt'
gt_car_id = 1  # 您想要提取的gt文件中的car_id
pred_car_id = 5  # 您想要提取的pred文件中的car_id
output_dir = "测试"

extract_and_save_data(gt_file, pred_file, gt_car_id, pred_car_id, output_dir)
