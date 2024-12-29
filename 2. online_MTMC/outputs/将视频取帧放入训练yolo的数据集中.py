# /home/chatmindai/project/zhangkun/Fast_Online_MTMCT/2. online_MTMC/yolov10/datasets2/multi_class
# 视频路径：/home/chatmindai/project/zhangkun/Fast_Online_MTMCT/dataset/HST/real
# test_gt.txt
# 根据视频，还有test_gt.txt，将视频中的帧提取出来，放入到multi_class中 所有的帧，分成11份，10分放入train文件夹，1份放入val文件夹，以coco格式

import os
import cv2
import json
import random
from tqdm import tqdm

# 定义路径
base_video_path = "/home/chatmindai/project/zhangkun/Fast_Online_MTMCT/dataset/AIC19/validation/S02/"
gt_file = "/home/chatmindai/project/zhangkun/Fast_Online_MTMCT/2. online_MTMC/outputs/ground_truth_validation.txt"  # 请确保此路径正确
output_path = "/home/chatmindai/project/zhangkun/Fast_Online_MTMCT/2. online_MTMC/yolov10/datasets2/multi_class"

# 创建输出目录
os.makedirs(os.path.join(output_path, "train", "images"), exist_ok=True)
os.makedirs(os.path.join(output_path, "train", "labels"), exist_ok=True)
os.makedirs(os.path.join(output_path, "val", "images"), exist_ok=True)
os.makedirs(os.path.join(output_path, "val", "labels"), exist_ok=True)

# 读取test_gt.txt文件
def read_gt_file(file_path):
    annotations = {}
    with open(file_path, 'r') as f:
        for line in f:
            camera_id,  track_id, frame, left, top, width, height, _, _ = line.strip().split(' ')
            camera_id = int(camera_id)
            frame = int(frame)
            if camera_id not in annotations:
                annotations[camera_id] = {}
            if frame not in annotations[camera_id]:
                annotations[camera_id][frame] = []
            annotations[camera_id][frame].append({
                "track_id": int(track_id),
                "bbox": [left, top, width, height]
            })
    return annotations

# 将边界框转换为COCO格式
def convert_to_coco(bbox, img_width, img_height):
    x, y, w, h = bbox
    x = int(x)
    y = int(y)
    w = int(w)
    h = int(h)

    x_center = (x + w / 2) / img_width
    y_center = (y + h / 2) / img_height
    w = w / img_width
    h = h / img_height
    return [x_center, y_center, w, h]

# 主处理函数
def process_videos():
    annotations = read_gt_file(gt_file)
    all_frames = []

    for camera_id in os.listdir(base_video_path):
        camera_path = os.path.join(base_video_path, camera_id)
        if os.path.isdir(camera_path):
            video_file = f"vdo.avi"
            video_path = os.path.join(camera_path, video_file)
            if os.path.exists(video_path):
                video = cv2.VideoCapture(video_path)
                frame_count = 0

                while True:
                    ret, frame = video.read()
                    if not ret:
                        break

                    id_ = int(camera_id[-1:])
                    if id_ in annotations and frame_count in annotations[id_]:
                        frame_filename = f"{id_}_{frame_count:06d}.jpg"
                        all_frames.append((frame, frame_filename, annotations[id_][frame_count]))
                    frame_count += 1
                video.release()

    # 打乱帧的顺序并分割为训练集和验证集
    random.shuffle(all_frames)
    split_index = len(all_frames) // 11 * 10
    train_frames = all_frames[:split_index]
    val_frames = all_frames[split_index:]

    # 处理训练集
    process_dataset(train_frames, "train")

    # 处理验证集
    process_dataset(val_frames, "val")

def process_dataset(frames, dataset_type):
    for frame, filename, frame_annotations in tqdm(frames, desc=f"处理{dataset_type}集"):
        # 保存图像
        cv2.imwrite(os.path.join(output_path, dataset_type, "images", filename), frame)

        # 创建标签文件
        img_height, img_width = frame.shape[:2]
        label_content = ""
        for ann in frame_annotations:
            bbox_coco = convert_to_coco(ann['bbox'], int(img_width), int(img_height))
            label_content += f"2 {' '.join(map(str, bbox_coco))}\n"

        # 保存标签文件
        with open(os.path.join(output_path, dataset_type, "labels", filename.replace('.jpg', '.txt')), 'w') as f:
            f.write(label_content)

if __name__ == "__main__":
    process_videos()
