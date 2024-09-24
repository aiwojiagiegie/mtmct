import cv2
import numpy as np
from tqdm import tqdm
import os
from concurrent.futures import ThreadPoolExecutor

def hsv2bgr(h, s, v):
    h_i = int(h * 6)
    f = h * 6 - h_i
    p = v * (1 - s)
    q = v * (1 - f * s)
    t = v * (1 - (1 - f) * s)

    r, g, b = 0, 0, 0

    if h_i == 0:
        r, g, b = v, t, p
    elif h_i == 1:
        r, g, b = q, v, p
    elif h_i == 2:
        r, g, b = p, v, t
    elif h_i == 3:
        r, g, b = p, q, v
    elif h_i == 4:
        r, g, b = t, p, v
    elif h_i == 5:
        r, g, b = v, p, q

    return int(b * 255), int(g * 255), int(r * 255)

def random_color(id):
    h_plane = (((id << 2) ^ 0x937151) % 100) / 100.0
    s_plane = (((id << 3) ^ 0x315793) % 100) / 100.0
    return hsv2bgr(h_plane, s_plane, 1)

def get_bbox_data(txt_path):
    # 读取TXT文件中的bbox信息
    bbox_data = {}
    car_bbox_colors = {}
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
            if car_id not in car_bbox_colors:
                car_bbox_colors[car_id] = random_color(car_id)
            if camera_id not in bbox_data:
                bbox_data[camera_id] = {}
            if frame_number not in bbox_data[camera_id]:
                bbox_data[camera_id][frame_number] = []
            bbox_data[camera_id][frame_number].append((left, top, width, height, car_id))
    return bbox_data, car_bbox_colors

def process_video_with_optical_flow(input_path, output_path_overlay, output_path_flow, mask_path, gt_file_path, cam):
    # 确保输出文件夹存在
    os.makedirs(os.path.dirname(output_path_overlay), exist_ok=True)
    os.makedirs(os.path.dirname(output_path_flow), exist_ok=True)

    cap = cv2.VideoCapture(input_path)
    
    # 获取视频属性
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # 读取掩码图片
    mask_img = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    _, mask = cv2.threshold(mask_img, 127, 255, cv2.THRESH_BINARY)

    # 读取GT文件中的bbox信息
    bbox_info, car_bbox_colors = get_bbox_data(gt_file_path)
    bbox_data_gt = bbox_info.get(cam, {}).get('gt', {})

    # 定义视频写入对象
    out_overlay = cv2.VideoWriter(output_path_overlay, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))
    out_flow = cv2.VideoWriter(output_path_flow, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

    # 读取第一帧
    ret, old_frame = cap.read()
    old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
    old_gray_masked = cv2.bitwise_and(old_gray, old_gray, mask=mask)

    progress_bar = tqdm(total=total_frames, desc=f"处理{input_path}中")

    frame_number = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break  # 视频结束

        frame_number += 1
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame_gray_masked = cv2.bitwise_and(frame_gray, frame_gray, mask=mask)

        # 计算光流
        flow = cv2.calcOpticalFlowFarneback(old_gray_masked, frame_gray_masked, None, 0.5, 3, 15, 3, 5, 1.2, 0)

        # 将光流转换为 HSV 图像进行可视化
        hsv = np.zeros_like(frame)
        hsv[..., 1] = 255

        # 计算光流的角度和幅度
        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        
        # 设置运动幅度阈值
        motion_threshold = 4.0  # 根据实际情况调整这个值
        mag[mag < motion_threshold] = 0
        
        hsv[..., 0] = ang * 180 / np.pi / 2  # 角度映射到 [0, 180] 的范围
        hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)  # 幅度归一化到 [0, 255]

        # 将 HSV 转换为 BGR 图像
        bgr_flow = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

        # 创建遮罩，只显示有明显运动的区域
        motion_mask = (hsv[..., 2] > 20).astype(np.uint8) * 255
        motion_mask = cv2.bitwise_and(motion_mask, motion_mask, mask=mask)

        # 对遮罩进行形态学操作，去除噪点
        kernel = np.ones((5,5), np.uint8)
        motion_mask = cv2.morphologyEx(motion_mask, cv2.MORPH_OPEN, kernel)
        motion_mask = cv2.morphologyEx(motion_mask, cv2.MORPH_CLOSE, kernel)

        # 寻找连通区域
        contours, _ = cv2.findContours(motion_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # 创建一个新的遮罩，只保留大区域
        filtered_motion_mask = np.zeros_like(motion_mask)

        # 在原始帧和光流结果上绘制边界框
        for contour in contours:
            # 计算边界框
            x, y, w, h = cv2.boundingRect(contour)
            
            # 过滤掉太小的框
            if w * h > 400:
                # 在原始帧上绘制
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                
                # 在光流结果上绘制
                cv2.rectangle(bgr_flow, (x, y), (x+w, y+h), (0, 255, 0), 2)
                
                # 将大区域添加到新的遮罩中
                cv2.drawContours(filtered_motion_mask, [contour], -1, 255, thickness=cv2.FILLED)

        # 绘制GT文件中的bbox
        if frame_number in bbox_data_gt:
            for bbox in bbox_data_gt[frame_number]:
                left, top, width, height, car_id = bbox
                # 绘制边界框
                cv2.rectangle(frame, (left, top), (left + width, top + height), car_bbox_colors[car_id], 2)
                # 绘制文本
                cv2.putText(frame, f'gt_id: {car_id}', (left, top - 5), 0, 1, (255, 255, 255), 2, 16)

        # 将光流结果叠加到原始帧上，但不使用遮罩
        overlay = cv2.addWeighted(frame, 1, bgr_flow, 0.7, 0)
        result = cv2.add(frame, cv2.bitwise_and(overlay, overlay, mask=filtered_motion_mask))

        # 写入输出视频
        out_overlay.write(result)
        out_flow.write(bgr_flow)

        # 更新前一帧
        old_gray_masked = frame_gray_masked.copy()
        progress_bar.update(1)

    # 释放资源
    cap.release()
    out_overlay.release()
    out_flow.release()
    cv2.destroyAllWindows()
    progress_bar.close()

def process_folder(folder_number, gt_file_path):
    input_video = f'dataset/HST/real/{folder_number}/{folder_number}.mp4'
    output_overlay = f'dataset/HST/real/{folder_number}/v{version}/光流视频重叠_masked.mp4'
    output_flow = f'dataset/HST/real/{folder_number}/v{version}/仅光流_masked.mp4'
    mask_path = f'dataset/HST/real/{folder_number}/{folder_number}.png'
    
    process_video_with_optical_flow(input_video, output_overlay, output_flow, mask_path, gt_file_path, folder_number)

# 主程序
if __name__ == "__main__":
    gt_file_path = '2. online_MTMC/output_HST/test_gt.txt'
    folders = range(41, 47)  # 41-46
    version = 6
    with ThreadPoolExecutor(max_workers=6) as executor:
        executor.map(lambda folder: process_folder(folder, gt_file_path), folders)

    print("所有视频处理完成")
