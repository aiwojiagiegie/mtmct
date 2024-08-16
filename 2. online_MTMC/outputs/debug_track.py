import numpy as np
import os
import sys

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


def get_track_data(txt_path):
    # 读取TXT文件中的bbox信息
    track_data = {}
    car_bbox_colors = {}
    with open(txt_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 0:
                break
            camera_id = int(parts[0])
            frame_number = int(parts[2])
            track_id = int(parts[1])
            left = int(parts[3])
            top = int(parts[4])
            width = int(parts[5])
            height = int(parts[6])
            if track_id not in car_bbox_colors:
                car_bbox_colors[track_id] = random_color(track_id)
            if camera_id not in track_data:
                track_data[camera_id] = {}
            if track_id not in track_data[camera_id]:
                track_data[camera_id][track_id] = []
            track_data[camera_id][track_id].append([frame_number, left, top, width, height])
    return track_data, car_bbox_colors

def are_tracks_same(track1, track2):
    ious = []
    for frame1 in track1:
        for frame2 in track2:
            if frame1[0] == frame2[0]:  # Compare frames with the same frame_number
                iou = calculate_iou(frame1[1:], frame2[1:])
                ious.append(iou)
                break
    return all(iou > 0.5 for iou in ious) if ious else False


if __name__ == '__main__':
    gt_path = 'ground_truth_validation.txt'
    detection_path = 'result/v1.txt'
    gt_track, _ = get_track_data(gt_path)
    detection_track, _ = get_track_data(detection_path)
    track1 = [[1, 100, 100, 50, 50], [2, 150, 150, 50, 50]]
    track2 = [[1, 105, 105, 50, 50], [2, 155, 150, 50, 50]]
    # for e in gt_track:
    #     for f in detection_track:

