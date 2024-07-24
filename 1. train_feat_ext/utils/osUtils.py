import cv2
import os

def video_to_frames(video_path, output_folder,camera_id):
    # 创建输出文件夹
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 打开视频文件
    cap = cv2.VideoCapture(video_path)

    # 检查是否成功打开
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    # 获取视频的帧率
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"Frames per second using video.get(cv2.CAP_PROP_FPS): {fps:.2f}")

    # 初始化帧数计数器
    frame_count = 0

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        # 构建图片路径
        img_path = os.path.join(output_folder, f'../frame/{camera_id}_f{frame_count+1:04d}.jpg')

        # 写入图片
        cv2.imwrite(img_path, frame)
        print(f'写入图片{img_path}')
        # 增加帧数计数器
        frame_count += 1

    # 释放资源
    cap.release()
    print(f"Total frames extracted: {frame_count}")
def get_video_list(root_dir):
    video_extensions = ('.mp4', '.avi', '.mov', '.mkv', '.wmv')  # 视频文件扩展名
    video_list = []

    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if file.endswith(video_extensions):
                video_list.append(os.path.join(root, file))

    return video_list
if __name__ == '__main__':
    mtmct_dataset = '/data2/zhangkun/project/Fast_Online_MTMCT/dataset/AIC19/validation/S02'
    video_list = get_video_list(mtmct_dataset)
    for video_path in video_list:
        parts = video_path.split('/')
        camera_id = parts[-2]  # 取倒数第三部分，即'c006'
        video_to_frames(video_path,f'../../dataset/AIC19/validation/S02/{camera_id}/frame',camera_id)
