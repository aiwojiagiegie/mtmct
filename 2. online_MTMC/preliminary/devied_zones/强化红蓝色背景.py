import cv2
import numpy as np

def enhance_red_blue(image):
    # 将图像转换为RGB颜色空间
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # 定义红色和蓝色的阈值
    red_threshold = 200
    blue_threshold = 200
    
    # 创建红色和蓝色的掩码
    red_mask = (rgb_image[:,:,0] > red_threshold) & (rgb_image[:,:,1] < 120) & (rgb_image[:,:,2] < 120)
    blue_mask = (rgb_image[:,:,2] > blue_threshold) & (rgb_image[:,:,0] < 120) & (rgb_image[:,:,1] < 120)
    
    # 强化红色和蓝色
    rgb_image[red_mask] = [255, 0, 0]  # 纯红色
    rgb_image[blue_mask] = [0, 0, 255]  # 纯蓝色
    
    # 将图像转换回BGR颜色空间
    enhanced_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
    
    return enhanced_image

if __name__ == "__main__":
    image = cv2.imread("./41.png")
    enhanced = enhance_red_blue(image)
    cv2.imwrite("./41.png", enhanced)

    image = cv2.imread("./42.png")
    enhanced = enhance_red_blue(image)
    cv2.imwrite("./42.png", enhanced)

    image = cv2.imread("./43.png")
    enhanced = enhance_red_blue(image)
    cv2.imwrite("./43.png", enhanced)

    image = cv2.imread("./44.png")
    enhanced = enhance_red_blue(image)
    cv2.imwrite("./44.png", enhanced)

    image = cv2.imread("./45.png")
    enhanced = enhance_red_blue(image)
    cv2.imwrite("./45.png", enhanced)

    image = cv2.imread("./46.png")
    enhanced = enhance_red_blue(image)
    cv2.imwrite("./46.png", enhanced)

