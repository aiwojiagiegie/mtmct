import cv2
import numpy as np
from PIL import Image

def analyze_color_distribution(image_path):
    # 读取图像
    img = cv2.imread(image_path)
    
    # 转换为RGB颜色空间
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # 创建非黑色像素的掩码
    non_black_mask = np.any(img_rgb != [0, 0, 0], axis=-1)
    
    # 获取非黑色像素
    non_black_pixels = img_rgb[non_black_mask]
    
    # 计算总的非黑色像素数
    total_non_black = non_black_pixels.shape[0]
    
    # 计算纯色区域的比例
    pure_red = np.sum(np.all(non_black_pixels == [255, 0, 0], axis=-1))
    pure_red = pure_red / total_non_black
    pure_green = np.sum(np.all(non_black_pixels == [0, 255, 0], axis=-1))
    pure_green = pure_green / total_non_black
    pure_blue = np.sum(np.all(non_black_pixels == [0, 0, 255], axis=-1))
    pure_blue = pure_blue / total_non_black
    pure_white = np.sum(np.all(non_black_pixels == [255, 255, 255], axis=-1))
    pure_white = pure_white / total_non_black
    
    return {
        "纯红色占比": pure_red,
        "纯绿色占比": pure_green,
        "纯蓝色占比": pure_blue,
        "纯白色占比": pure_white
    }
if __name__ == "__main__":  
    # 使用示例
    image_path = "./44.png"
    result = analyze_color_distribution(image_path)
    for color, ratio in result.items():
        print(f"{color}: {ratio:.2%}")
