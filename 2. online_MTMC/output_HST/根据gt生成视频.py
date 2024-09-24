# 根据gt文件的数据，生成视频
import os
import cv2
from tqdm import tqdm

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

# 根据gt生成视频
def generate_video_from_gt(gt_file_path, video_path, output_path, cam):
    bbox_info, car_bbox_colors = get_bbox_data(gt_file_path)
    all_in_bbox_info = {}
    for camera_id, frame_bboxes in bbox_info.items():
        if camera_id not in all_in_bbox_info:
            all_in_bbox_info[camera_id] = {}
        all_in_bbox_info[camera_id]['gt'] = frame_bboxes
    # 读取视频
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Cannot open video.")
        return
    # 获取视频的基本信息
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    # 获取视频的总帧数
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 输出视频的编码

    directory = os.path.dirname(output_path)
    if not os.path.exists(directory):
        os.makedirs(directory)

    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # 读取每一帧，并绘制边界框
    frame_number = 0
    # 创建tqdm进度条
    progress_bar = tqdm(total=total_frames, desc=f"处理{video_path}中")
    bbox_data_gt = all_in_bbox_info[cam]['gt']
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_number += 1

        # 检查当前帧是否有bbox数据
        if frame_number in bbox_data_gt:
            for bbox in bbox_data_gt[frame_number]:
                left, top, width, height, car_id = bbox
                # 绘制边界框
                cv2.rectangle(frame, (left, top), (left + width, top + height), car_bbox_colors[car_id], 2)
                # 绘制文本
                cv2.putText(frame, f'gt_id: {car_id}', (left, top - 5), 0, 1, (255, 255, 255), 2, 16)

        # 将帧写入输出视频
        out.write(frame)
        progress_bar.update(1)

    # 释放资源
    cap.release()
    out.release()
    print("Video processing complete, saved to", output_path)

if __name__ == '__main__':
    gt_file_path = '2. online_MTMC/output_HST/test_gt.txt'
    video_dir_path = 'dataset/HST/real'
    # 遍历video_dir_path文件夹下的所有文件夹
    cams=[41,42,43,44,45,46]
    for cam in cams:
        video_path = os.path.join(video_dir_path, str(cam), f'{cam}.mp4')
        output_path = os.path.join('2. online_MTMC/output_HST/根据gt文件生成gt的debug视频/', f'{cam}.mp4')
        generate_video_from_gt(gt_file_path, video_path, output_path , cam)

