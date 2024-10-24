import os
import shutil
import sys
from concurrent.futures import ThreadPoolExecutor
import threading
from opts import opt
import cv2
import numpy as np
from tqdm import tqdm
import subprocess

sys.path.append('../')
from utils.IouUtils import calculate_iou

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


def draw_bboxes(video_path, bbox_data_gt, output_path, car_bbox_colors):
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
                cv2.putText(frame, f'id: {car_id}', (left, top - 5), 0, 1, (0, 0, 0), 2, 16)

        # 将帧写入输出视频
        out.write(frame)
        progress_bar.update(1)

    # 释放资源
    cap.release()
    out.release()
    print("Video processing complete, saved to", output_path)

def draw_bboxes_single_in(video_path, bbox_data_gt, output_path, car_bbox_colors):
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

def draw_bboxes_all_in(video_path, bbox_data_gt, bbox_data_pred, output_path, car_bbox_colors, pred_bbox_colors):
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

        if frame_number in bbox_data_pred:
            for bbox in bbox_data_pred[frame_number]:
                left, top, width, height, car_id = bbox
                # 绘制边界框
                cv2.rectangle(frame, (left, top), (left + width, top + height), pred_bbox_colors[car_id], 2)
                # 绘制文本
                cv2.putText(frame, f'pred_id: {car_id}', (left, top + height - 5), 0, 1, (255, 255, 255), 2, 16)

        # 将帧写入输出视频
        out.write(frame)
        progress_bar.update(1)

    # 释放资源
    cap.release()
    out.release()
    print("Video processing complete, saved to", output_path)


def draw_bboxes_remove_duplicate(video_path, bbox_data_gt, bbox_data_pred, output_path, gt_bbox_colors,
                                 pred_bbox_colors):
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
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_number += 1
        pred_ious = []
        gt_ious = []
        if frame_number in bbox_data_gt:
            for bbox in bbox_data_gt[frame_number]:
                left, top, width, height, car_id = bbox
                gt_ious.append(bbox)
        if frame_number in bbox_data_pred:
            for bbox in bbox_data_pred[frame_number]:
                left, top, width, height, car_id = bbox
                pred_ious.append(bbox)
        new_gt_ious = filter_by_ious(gt_ious, pred_ious)
        new_pred_ious = filter_by_ious(pred_ious, gt_ious)
        # ious = [i for i in new_ious if i[4] == False]
        for bbox in new_gt_ious:
            left, top, width, height, car_id = bbox
            # 绘制边界框
            cv2.rectangle(frame, (left, top), (left + width, top + height), gt_bbox_colors[car_id], 2)
            # 绘制文本
            cv2.putText(frame, f'gt_id: {car_id}', (left, top + 5), 0, 1, (255, 255, 255), 2, 16)
        # for bbox in new_pred_ious:
        #     left, top, width, height, car_id = bbox
        #     # 绘制边界框
        #     cv2.rectangle(frame, (left, top), (left + width, top + height), pred_bbox_colors[car_id], 2)
        #     # 绘制文本
        #     cv2.putText(frame, f'pred_id: {car_id}', (left, top + 5), 0, 1, (255, 255, 255), 2, 16)
        # 将帧写入输出视频
        out.write(frame)
        progress_bar.update(1)

    # 释放资源
    cap.release()
    out.release()
    print("Video processing complete, saved to", output_path)


def filter_by_ious(ious, another_ious):
    new_ious = []
    for i in ious:
        remove_flag = False
        for j in another_ious:
            if i is not j and calculate_iou(i, j) > 0.5:
                remove_flag = True
                break
        if not remove_flag:
            new_ious.append(i)
    return new_ious


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


def generate_depart():
    '''
    生成可视化视频文件，分成两部分
    '''
    bbox_info, car_bbox_colors = get_bbox_data(gt_path)
    for camera_id, frame_bboxes in bbox_info.items():
        draw_bboxes(f'../../dataset/AIC19/validation/S02/c00{camera_id}/vdo.avi', frame_bboxes,
                    f'debug/gt/c00{camera_id}.mp4', car_bbox_colors)
    target_info, car_bbox_colors = get_bbox_data(detection_path)
    for camera_id, frame_bboxes in target_info.items():
        draw_bboxes(f'../../dataset/HST/real/{camera_id}/{camera_id}.mp4',
                    frame_bboxes,
                    f'debug/{version}/target/{camera_id}.mp4', car_bbox_colors)
    # 启动新的一个线程去执行下面这个函数
    thread = threading.Thread(target=compress_video, args=(f'debug/{version}/target/',))
    thread.start()


def generate_all_in():
    bbox_info, gt_bbox_colors = get_bbox_data(gt_path)
    target_info, pred_bbox_colors = get_bbox_data(detection_path)
    all_in_bbox_info = {}
    for camera_id, frame_bboxes in bbox_info.items():
        if camera_id not in all_in_bbox_info:
            all_in_bbox_info[camera_id] = {}
        all_in_bbox_info[camera_id]['gt'] = frame_bboxes
    for camera_id, frame_bboxes in target_info.items():
        all_in_bbox_info[camera_id]['pred'] = frame_bboxes
    
    for camera_id, frame_bboxes in all_in_bbox_info.items():
        output_path = f'debug/{version}/allIn/{camera_id}.mp4'
        draw_bboxes_all_in(f'../../dataset/HST/real/{camera_id}/{camera_id}.mp4',
                           frame_bboxes['gt'],
                           frame_bboxes['pred'],
                           output_path, gt_bbox_colors, pred_bbox_colors)
        
        # 为每个生成的视频启动一个新线程进行压缩
        thread = threading.Thread(target=compress_single_video, args=(output_path,))
        thread.start()

def compress_single_video(input_path):
    output_dir = os.path.join(os.path.dirname(input_path), '压缩')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    output_path = os.path.join(output_dir, os.path.basename(input_path))
    
    command = [
        "ffmpeg",
        "-i", input_path,
        "-c:v", "libx264",
        "-crf", '23',
        "-preset", 'medium',
        "-c:a", "copy",
        output_path
    ]

    try:
        subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print(f"视频 {input_path} 压缩完成!")
    except subprocess.CalledProcessError as e:
        print(f"压缩 {input_path} 时出错: {e}")

def generate_remove_duplicate():
    bbox_info, gt_bbox_colors = get_bbox_data(gt_path)
    target_info, pred_bbox_colors = get_bbox_data(detection_path)
    all_in_bbox_info = {}
    for camera_id, frame_bboxes in bbox_info.items():
        if camera_id not in all_in_bbox_info:
            all_in_bbox_info[camera_id] = {}
        all_in_bbox_info[camera_id]['gt'] = frame_bboxes
    for camera_id, frame_bboxes in target_info.items():
        all_in_bbox_info[camera_id]['pred'] = frame_bboxes
    for camera_id, frame_bboxes in all_in_bbox_info.items():
        draw_bboxes_remove_duplicate(f'../../dataset/HST/real/{camera_id}/{camera_id}.mp4',
                                     frame_bboxes['gt'],
                                     frame_bboxes['pred'],
                                     f'debug/{version}/remove_duplicate/{camera_id}.mp4', gt_bbox_colors, pred_bbox_colors)
    # 启动新的一个线程去执行下面这个函数
    thread = threading.Thread(target=compress_video, args=(f'debug/{version}/remove_duplicate/',))
    thread.start()

def compress_video(video_path):
    video_files = os.listdir(video_path)
    #video_files过滤出视频文件 并获取绝对路径
    video_files = [os.path.join(video_path, file) for file in video_files if file.endswith('.mp4')]
    output_path = os.path.join(video_path, '压缩')
    # 删除压缩文件夹
    if os.path.exists(output_path):
        shutil.rmtree(output_path)
    # 创建压缩文件夹
    if not os.path.exists(output_path):
        os.makedirs(output_path)    
    # 遍历并压缩所有视频
    for input_path in tqdm(video_files, desc="压缩视频"):
        
        # 使用FFmpeg进行压缩
        command = [
        "ffmpeg",
        "-i", input_path,
        "-c:v", "libx264",
        "-crf", '23',
        "-preset", 'medium',
        "-c:a", "copy",
        os.path.join(output_path,input_path.split('/')[-1])
        ]

        try:
            subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except subprocess.CalledProcessError as e:
            print(f"压缩 {input_path} 时出错: {e}")
        
    print("所有视频压缩完成!")
    


if __name__ == '__main__':
    # opt.version=25
    version = 'v'+str(opt.version)
    detection_path = f'result/version/{version}.txt'
    gt_path = 'test_gt.txt'
    generate_all_in()
    # generate_depart()
    # generate_remove_duplicate()
    # all_in_bbox_info = {}
    # for camera_id, frame_bboxes in bbox_info.items():
    #     if camera_id not in all_in_bbox_info:
    #         all_in_bbox_info[camera_id] = {}
    #     all_in_bbox_info[camera_id]['gt'] = frame_bboxes
    #
    # # 定义一个函数来处理单个camera_id的任务
    # def process_camera(camera_id, all_info):
    #     frame_bboxes = all_info[camera_id]
    #     draw_bboxes_single_in(f'../../dataset/HST/pred/{camera_id}.mp4', frame_bboxes['gt'],
    #                           f'debug/{version}/single/{camera_id}.mp4', gt_bbox_colors)
    #
    # # 创建线程池
    # with ThreadPoolExecutor(max_workers=6) as executor:
    #     # 提交任务给线程池
    #     futures = {executor.submit(process_camera, camera_id, all_in_bbox_info): camera_id for camera_id in
    #                all_in_bbox_info}

    # for camera_id, frame_bboxes in all_in_bbox_info.items():
    #     draw_bboxes_single_in(f'../../dataset/qianhuang/{camera_id}.mp4', frame_bboxes['gt'],
    #                        f'debug/{version}/single/{camera_id}.mp4', gt_bbox_colors)



