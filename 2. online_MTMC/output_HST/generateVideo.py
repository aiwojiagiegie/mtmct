import os

import cv2
import pandas as pd
"""
这个文件用于根据gt文件生成对应的视频，HST数据集的gt文件只标注了不到两分钟
"""
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
if __name__ == '__main__':
    gt_path = 'test_gt.txt'
    # detection_path = 'result/v1.txt'
    gt_data = pred_data = getData(gt_path, names)
    max_frameId = gt_data.groupby('CameraId')['FrameId'].max()
    max_frameid_dict = max_frameId.to_dict()
    for camera_id, frame_id in max_frameid_dict.items():
        video_path = f'../../dataset/HST/{camera_id}.mp4'
        # 打开视频文件
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            print(f"无法打开视频文件: {video_path}")
            continue
            # 获取视频的基本信息
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # 构造输出文件路径
        output_path = f"../../dataset/HST/pred/{camera_id}.mp4"
        directory = os.path.dirname(output_path)
        if not os.path.exists(directory):
            os.makedirs(directory)
        # 创建 VideoWriter 对象
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        # 读取视频帧并写入新文件
        for i in range(int(frame_id) + 1):
            ret, frame = cap.read()
            if not ret:
                break
            # 写入帧到输出文件
            out.write(frame)
        # 释放资源
        cap.release()
        out.release()
        print(f"已保存从第0帧到第{frame_id}帧的视频到文件: {output_path}")
    print(max_frameId)