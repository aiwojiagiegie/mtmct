import os
import sys

import cv2
import numpy as np
from tqdm import tqdm

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
                cv2.putText(frame, f'pred_id: {car_id}', (left, top + 5), 0, 1, (255, 255, 255), 2, 16)

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
        draw_bboxes(f'../../dataset/AIC19/validation/S02/c00{camera_id}/vdo.avi', frame_bboxes,
                    f'debug/target/c00{camera_id}.mp4', car_bbox_colors)


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
        draw_bboxes_all_in(f'../../dataset/AIC19/validation/S02/c00{camera_id}/vdo.avi', frame_bboxes['gt'],
                           frame_bboxes['pred'],
                           f'debug/allIn/c00{camera_id}.mp4', gt_bbox_colors, pred_bbox_colors)


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
        draw_bboxes_remove_duplicate(f'../../dataset/AIC19/validation/S02/c00{camera_id}/vdo.avi',
                                     frame_bboxes['gt'],
                                     frame_bboxes['pred'],
                                     f'debug/remove_duplicate/c00{camera_id}.mp4', gt_bbox_colors, pred_bbox_colors)


if __name__ == '__main__':
    detection_path = f'result/{MTMCT.mtmct_version}.txt'
    gt_path = 'ground_truth_validation.txt'
    generate_remove_duplicate()
