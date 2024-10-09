# 读取指定路径的视频, 根据视频名称生成key ，对应的value是视频的size
import os
import cv2
import numpy as np
size_dict = {}
def generate_images(video_path):
    # 打开视频文件
    video = cv2.VideoCapture(video_path)

    # 获取视频的宽度和高度
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    # key为视频名称，value为视频的size
    vidio_name = os.path.basename(video_path).split(".")[0]
    # 判断是否是数字
    if vidio_name.isdigit():
        size_dict[vidio_name] = (width, height)

if __name__ == "__main__":
    # 读取路径下面的所有视频
    for root, dirs, files in os.walk("dataset/HST/real"):
        for file in files:
            if file.endswith(".mp4"):
                video_path = os.path.join(root, file)
                generate_images(video_path)
    # 打印size_dict
    print(size_dict)
    # 在某个路径中生成key_key.png类似这样的图片，如果已经有了就不生成，图片大小为size_dict[key]，内容是全黑色的
    for key in size_dict:
        for key2 in size_dict:
            if key == key2:
                continue    
            output_path = os.path.join("2. online_MTMC/preliminary/overlap_zones/HST", key + "_" + key2 + ".png")
            if os.path.exists(output_path):
                continue
            img = np.zeros((size_dict[key][1], size_dict[key][0], 3), dtype=np.uint8)
            cv2.imwrite(output_path, img)
