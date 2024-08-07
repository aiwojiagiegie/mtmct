import os
import numpy as np
import pandas as pd
import cv2
from tqdm import tqdm

names = ['CameraId', 'Id', 'FrameId', 'X', 'Y', 'Width', 'Height', 'Xworld', 'Yworld']


def getData(fpath, names=None, sep='\s+|\t+|,'):
    try:
        df = pd.read_csv(
            fpath,
            sep=sep,
            index_col=None,
            skipinitialspace=True,
            header=None,
            names=names,
            engine='python'
        )
        return df
    except Exception as e:
        raise ValueError("Could not read input from %s. Error: %s" % (fpath, repr(e)))
cam_ids=[6,7,8,9]
caps={}
for cam_id in cam_ids:
    video_path = f"../../dataset/AIC19/validation/S02/c00{cam_id}/vdo.avi"
    if cam_id not in caps:
        caps[cam_id] = cv2.VideoCapture(video_path)
def process_data(df, vehicle_id, target_cam_ids, path='./all_in_pic/'):
    if not os.path.exists(path):
        os.makedirs(path)

    # 筛选特定车辆ID和摄像头ID
    selected_df = df[(df['Id'] == vehicle_id) & (df['CameraId'].isin(target_cam_ids))]
    if selected_df.empty:
        return

    max_height = 0
    max_width = 0
    images = []

    for cam_id in target_cam_ids:
        cap = caps[cam_id]
        for index, row in selected_df[selected_df['CameraId'] == cam_id].iterrows():
            cap.set(cv2.CAP_PROP_POS_FRAMES, row['FrameId'])
            ret, frame = cap.read()
            if not ret:
                print(f"Failed to retrieve frame from {video_path}")
                continue
            x, y, w, h = int(row['X']), int(row['Y']), int(row['Width']), int(row['Height'])
            crop_img = frame[y:y + h, x:x + w]
            max_height = max(max_height, h)
            max_width = max(max_width, w)
            images.append(crop_img)

    if not images:
        return

    adjusted_images = []
    for img in images:
        padding_top = (max_height - img.shape[0]) // 2
        padding_bottom = max_height - img.shape[0] - padding_top
        padding_left = (max_width - img.shape[1]) // 2
        padding_right = max_width - img.shape[1] - padding_left
        padded_img = cv2.copyMakeBorder(img, padding_top, padding_bottom, padding_left, padding_right,
                                        cv2.BORDER_CONSTANT)
        adjusted_images.append(padded_img)

    rows = 4  # 设定每行四个图像
    cols = (len(adjusted_images) + rows - 1) // rows  # 计算需要的行数
    matrix_height = max_height * rows
    matrix_width = max_width * cols
    final_matrix = np.zeros((matrix_height, matrix_width, 3), dtype=np.uint8)

    for i, img in enumerate(adjusted_images):
        start_row = (i // cols) * max_height
        start_col = (i % cols) * max_width
        final_matrix[start_row:start_row + max_height, start_col:start_col + max_width, :] = img

    if final_matrix.any():
        cv2.imwrite(f'{path}output_{vehicle_id}.jpg', final_matrix)  # 保存图像


if __name__ == '__main__':
    gt_path = 'ground_truth_validation.txt'
    detection_path = 'result/v1.txt'
    pred_data = getData(detection_path, names)
    unique_vehicle_ids = pred_data['Id'].unique()
    target_cam_ids = [6, 7, 8, 9]  # 目标摄像头ID
    for vehicle_id in tqdm(unique_vehicle_ids):
        process_data(pred_data, vehicle_id, target_cam_ids, path='./all_in_pic/pred/')
    for cap in caps:
        caps[cap].release()
