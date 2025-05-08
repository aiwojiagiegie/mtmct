import os

import numpy as np
from PIL import Image

def change_image_colors(root_folder, new_folder, color_sequence):
    # 定义颜色映射
    color_map = {
        'red': (255, 0, 0),
        'blue': (0, 0, 255),
        'original': (255, 255, 255)
    }

    # 创建新文件夹
    os.makedirs(new_folder, exist_ok=True)

    # 遍历每个文件夹
    for lane, images in color_sequence.items():
        folder_name = f'{lane}'
        folder_path = os.path.join(root_folder, folder_name)
        new_folder_path = os.path.join(new_folder, folder_name)
        os.makedirs(new_folder_path, exist_ok=True)

        # 遍历每个图片
        for image_num, color in images.items():
            image_name = f'{image_num}.jpg'
            image_path = os.path.join(folder_path, image_name)
            img = Image.open(image_path).convert('RGB')

            # 将图片转换为numpy数组
            img_array = np.array(img)
            
            # 创建白色像素的掩码
            white_mask = np.all(img_array >= [100, 100, 100], axis=2)
            
            # 使用掩码一次性替换所有白色像素
            target_color = np.array(color_map[color])
            img_array[white_mask] = target_color
            
            # 转回PIL图像并保存
            new_img = Image.fromarray(img_array)
            new_image_path = os.path.join(new_folder_path, image_name)
            new_img.save(new_image_path)

def main():
    # 根文件夹路径
    root_folder = 'AIC19_compressed'
    # 新文件夹路径
    new_folder = 'AIC19_2'

    # 定义颜色序列
    color_sequence = {
        1: {6: 'red', 7: 'original', 8: 'original', 9: 'blue'},
        2: {6: 'blue', 7: 'original', 8: 'original', 9: 'red'},
        3: {6: 'red', 7: 'blue', 8: 'original', 9: 'original'},
        4: {6: 'blue', 7: 'red', 8: 'original', 9: 'original'},
        5: {6: 'original', 7: 'original', 8: 'blue', 9: 'red'},
        6: {6: 'original', 7: 'blue', 8: 'original', 9: 'red'},
        7: {6: 'original', 7: 'red', 8: 'blue', 9: 'blue'},
        8: {6: 'original', 7: 'original', 8: 'red', 9: 'blue'},
        9: {6: 'original', 7: 'blue', 8: 'red', 9: 'original'},
        # 继续为其他车道定义颜色序列
    }

    change_image_colors(root_folder, new_folder, color_sequence)

if __name__ == "__main__":
    main()