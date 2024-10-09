# 读取指定路径的视频，根据视频的size生成对应大小的图片，并保存到指定路径，生成一张即可
import os
import cv2
import numpy as np

def generate_images(video_path, output_path,dir):
    # 打开视频文件
    video = cv2.VideoCapture(video_path)

    # 获取视频的宽度和高度
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # 创建输出目录 判断output_path有没有父目录，没有就创建
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # 读取视频的帧
    ret, frame = video.read()
    # 生成的图片全部都是黑色的
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    # 如果读取成功，保存图片
    if ret:
        # save_path = os.path.join(output_path, f"frame_{dir}.jpg")
        cv2.imwrite(output_path, frame)

    # 释放视频文件
    video.release()

if __name__ == "__main__":
    # 根据下面这个格式，首选读取路径下面的所有子路径，然后根据子路径列表for循环调用generate_images函数
    # 读取路径下面的所有子路径
    for root, dirs, files in os.walk("dataset/HST/real"):
        for dir in dirs:
            video_path = os.path.join(root, dir, dir + ".mp4")
            output_path = os.path.join("2. online_MTMC/preliminary/overlap_zones/HST", dir + ".png")
            generate_images(video_path, output_path,dir)
