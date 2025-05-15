import os
import cv2
from tqdm import tqdm
from yolov10.ultralytics import YOLOv10
import concurrent.futures

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

def process_video(video_path, output_path, gt_file_path, model):
    bbox_info, car_bbox_colors = get_bbox_data(gt_file_path)
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"无法打开视频文件: {video_path}")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    directory = os.path.dirname(output_path)
    if not os.path.exists(directory):
        os.makedirs(directory)
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    progress_bar = tqdm(total=total_frames, desc=f"处理{video_path}中")

    frame_number = 0
    cam = int(video_path.split('/')[-2].split('.')[0][1:])
    bbox_data_gt = bbox_info.get(cam, {})

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_number += 1

        # YOLOv10检测
        result = model(frame)[0]
        names = result.names
        boxes = result.boxes.data.tolist()
        for obj in boxes:
            left, top, right, bottom = map(int, obj[:4])
            cv2.rectangle(frame, (left, top), (right, bottom), (255,255,255), 2)
            label = f"{names[int(obj[5])]}: {obj[4]:.2f} | left:{left} top:{top} right:{right} bottom:{bottom}"
            cv2.putText(frame, label, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2)

        # 绘制GT边界框
        if frame_number in bbox_data_gt:
            for bbox in bbox_data_gt[frame_number]:
                left, top, width, height, car_id = bbox
                cv2.rectangle(frame, (left, top), (left + width, top + height), car_bbox_colors[car_id], 2)
                cv2.putText(frame, f'gt_id: {car_id}', (left, top + height - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        out.write(frame)
        progress_bar.update(1)

    cap.release()
    out.release()
    print(f"视频处理完成,已保存至 {output_path}")
def process_video_wrapper(args):
    video_path, output_path, gt_file_path, model = args
    process_video(video_path, output_path, gt_file_path, model)

if __name__ == '__main__':
    model = YOLOv10('/root/mtmct/2. online_MTMC/preliminary/det_weights/my/best_multiple3.pt')
    gt_file_path = '2. online_MTMC/outputs/ground_truth_validation.txt'
    video_dir_path = 'dataset/AIC19/validation/S02'
    output_dir_path = '2. online_MTMC/outputs/根据gt和目标检测模型生成合并在一起的视频'

    # 准备所有视频的处理参数
    video_tasks = []
    for cam in range(6, 10):
        video_path = os.path.join(video_dir_path,  f'c00{cam}/vdo.avi')
        output_path = os.path.join(output_dir_path, f'c00{cam}_combined.mp4')
        video_tasks.append((video_path, output_path, gt_file_path, model))

    # 使用线程池执行视频处理任务
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as executor:
        list(tqdm(executor.map(process_video_wrapper, video_tasks), total=len(video_tasks), desc="处理视频"))

    print("所有视频处理完成")
