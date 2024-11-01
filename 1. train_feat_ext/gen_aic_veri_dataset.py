import os
import cv2
import random

# Naming Rule of the bboxes
# In bbox "0001_c1s1_001051_00.jpg"
# "0001" is object ID
# "c1" is the camera ID
# "s1" is sequence ID of camera "1".
# "001051" is the 1051^th frame in the sequence "c1s1"
import shutil


def gen_aic_veri_dataset():
    # Create patches directory
    if not os.path.exists(crop_save):
        os.makedirs(crop_save)

    # For AIC191
    data_path = '../dataset/HST/'

    gt_list = open(
        '/home/chatmindai/project/zhangkun/Fast_Online_MTMCT/2. online_MTMC/output_HST/test_gt.txt').readlines()
    gt_dict = {}
    for gt_line in gt_list:
        gt_line = gt_line.split(' ')
        cam_num = str(gt_line[0])
        f_num = int(gt_line[2])
        obj_id = int(gt_line[1])
        left = int(gt_line[3])
        top = int(gt_line[4])
        w = int(gt_line[5])
        h = int(gt_line[6])
        gt_dict[(cam_num, f_num, obj_id)] = (left, top, w, h)

    # 创建三个目标文件夹
    data_paths = ['../dataset/VeRi/image_query/', '../dataset/VeRi/image_test/', '../dataset/VeRi/image_train/']
    for path in data_paths:
        if not os.path.exists(path):
            os.makedirs(path)

    # 创建name_*.txt文件
    name_files = {
        'query': open('../dataset/VeRi/name_query.txt', 'a'),
        'test': open('../dataset/VeRi/name_test.txt', 'a'),
        'train': open('../dataset/VeRi/name_train.txt', 'a')
    }

    # 获取所有唯一的obj_id
    all_obj_ids = set(key[2] for key in gt_dict.keys())
    total_ids = len(all_obj_ids)

    # 随机分配obj_id到三个集合
    train_ids = set(random.sample(all_obj_ids, int(total_ids * 0.6)))
    remaining_ids = all_obj_ids - train_ids
    query_ids = set(random.sample(remaining_ids, int(total_ids * 0.15)))
    test_ids = remaining_ids - query_ids

    # 创建obj_id到目标路径的映射
    id_to_path = {}
    for obj_id in train_ids:
        id_to_path[obj_id] = data_paths[2]  # train
    for obj_id in query_ids:
        id_to_path[obj_id] = data_paths[0]  # query
    for obj_id in test_ids:
        id_to_path[obj_id] = data_paths[1]  # test

    scenes = ['real']
    for scene in scenes:
        # 对于每个摄像头
        cams = os.listdir(data_path + scene)
        for cam in cams:
            # 设置视频路径
            video_path = os.path.join(data_path, scene, cam, f'{cam}.mp4')  # 假设视频文件名为video.mp4
            
            # 打开视频流
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                print(f"无法打开视频: {video_path}")
                continue
                
            frame_count = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                    
                # 处理当前帧的所有bbox
                current_frame_keys = [k for k in gt_dict.keys() if k[0] == cam and k[1] == frame_count]
                
                for key in current_frame_keys:
                    # 读取GT
                    cam_num, f_num, obj_id = key
                    left, top, w, h = gt_dict[key]
                    
                    # 裁剪并保存bbox
                    bbox = frame[top:top + h + 1, left:left + w + 1, :]
                    target_path = id_to_path[obj_id]
                    image_name = '%04d_%s_%08d_0.jpg' % (obj_id + 1000, cam, f_num + 1)
                    image_path = os.path.join(target_path, image_name)
                    cv2.imwrite(image_path, bbox)
                    
                    # 写入对应的name_*.txt文件
                    if target_path == data_paths[0]:
                        name_files['query'].write(image_name + '\n')
                    elif target_path == data_paths[1]:
                        name_files['test'].write(image_name + '\n')
                    else:
                        name_files['train'].write(image_name + '\n')
                    
                    print(f'完成生成{image_path}')
                
                frame_count += 1
            
            # 释放视频资源
            cap.release()
            print('%s_%s 已完成' % (scene, cam))

    # 关闭name_*.txt文件
    for file in name_files.values():
        file.close()

    print(f"训练集大小: {len(train_ids)}")
    print(f"查询集大小: {len(query_ids)}")
    print(f"测试集大小: {len(test_ids)}")


def transfer():
    # For VeRi
    data_paths = ['../dataset/VeRi/image_query/', '../dataset/VeRi/image_test/', '../dataset/VeRi/image_train/']

    for data_path in data_paths:
        if not os.path.exists(data_path):
            os.makedirs(data_path)
        img_names = os.listdir(data_path)
        for img_name in img_names:
            obj_id, cam, f_num, _ = img_name.split('.jpg')[0].split('_')
            new_img_name = '%04d_c%03d_%s_1.jpg' % (int(obj_id) + 183, int(cam[1:]) + 40, f_num)
            old_image_path = data_path + img_name
            new_image_path = crop_save + new_img_name
            shutil.copy(old_image_path, new_image_path)


if __name__ == '__main__':
    crop_save = '../dataset/VeRi/'
    gen_aic_veri_dataset()
