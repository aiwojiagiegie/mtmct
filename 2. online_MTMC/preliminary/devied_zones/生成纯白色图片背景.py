import cv2
import numpy as np

def process_image(input_path, output_path):
    # 读取图片
    img = cv2.imread(input_path)
    
    # 检查图片是否成功读取
    if img is None:
        print(f"无法读取图片: {input_path}")
        return
    
    # 创建一个掩码,标记需要变成白色的像素
    mask = np.logical_or(
        (img[:,:,2] > 100),  # 红色通道 > 100
        (img[:,:,0] > 100)   # 蓝色通道 > 100
    )
    
    # 将满足条件的像素设置为白色
    img[mask] = [255, 255, 255]
    
    # 将图像转换为灰度图
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 保存处理后的8位灰度图片
    cv2.imwrite(output_path, gray_img)
    print(f"处理后的8位灰度图片已保存至: {output_path}")

# 使用示例
process_image("./41.png", "../rois/41.png")
process_image("./42.png", "../rois/42.png")
process_image("./43.png", "../rois/43.png")
process_image("./44.png", "../rois/44.png")
process_image("./45.png", "../rois/45.png")
process_image("./46.png", "../rois/46.png")
