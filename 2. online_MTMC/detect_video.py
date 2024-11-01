import os

import torch
import torchvision.transforms as transforms
from tqdm import tqdm

from yolov10.ultralytics import YOLOv10

import cv2

def read_video_frames(video_path,output_path):
    # 打开视频文件
    cap = cv2.VideoCapture(video_path)

    # 检查视频是否成功打开
    if not cap.isOpened():
        print(f"无法打开视频文件: {video_path}")
        return
    # 获取视频的总帧数
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    progress_bar = tqdm(total=total_frames, desc=f"处理{video_path}中")

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 输出视频的编码
    # 获取视频的基本信息
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    directory = os.path.dirname(output_path)
    if not os.path.exists(directory):
        os.makedirs(directory)
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    while True:
        # 读取一帧
        ret, frame = cap.read()        
        if not ret:
            break
        
        # 进行检测
        result = model(frame)[0]

        names = result.names
        boxes = result.boxes.data.tolist()
        for obj in boxes:
            # left, top, right, bottom = obj[:4]
            left, top, right, bottom = int(obj[0]), int(obj[1]), int(obj[2]), int(obj[3])
            # 将检测到的目标框绘制在图像上
            cv2.rectangle(frame, (left, top), (right, bottom), (255,255,255), 2)
            # 在图像上绘制目标的类别和置信度
            label = f"{names[int(obj[5])]}: {obj[4]:.2f}"
            cv2.putText(frame, label, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2)
        
        out.write(frame)
        progress_bar.update(1)

    # 释放资源
    cap.release()
    out.release()
    print(f'结束处理视频{video_path}')


if __name__ == '__main__':
    model = YOLOv10('/home/chatmindai/project/zhangkun/yolov10/runs/detect/UA-DETRAC_pre/model_name_yolov10s.pt/epochs_200/batch_327/weights/best.pt')
    for i in range(41,47):
        video_path = f"/home/chatmindai/project/zhangkun/Fast_Online_MTMCT/dataset/HST/real/{i}/{i}.mp4"
        read_video_frames(video_path,f'/home/chatmindai/project/zhangkun/Fast_Online_MTMCT/2. online_MTMC/output_HST/debug/往前偏移1帧训练的模型/{i}.mp4')
