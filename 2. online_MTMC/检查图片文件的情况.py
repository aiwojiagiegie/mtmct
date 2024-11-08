import cv2
import numpy as np

def get_pixel_color(image_path, x, y):
    """
    读取图片中指定坐标的颜色值
    
    Args:
        image_path (str): 图片路径
        x (int): x坐标
        y (int): y坐标
    
    Returns:
        tuple: BGR颜色值和灰度值
    """
    # 读取图片
    image = cv2.imread(image_path)
    
    # 检查图片是否成功读取
    if image is None:
        raise ValueError(f"无法读取图片: {image_path}")
    
    # 获取图片尺寸
    height, width = image.shape[:2]
    print(f"图片尺寸: 宽={width}, 高={height}")
    
    # 检查坐标是否有效
    if x < 0 or x >= width or y < 0 or y >= height:
        raise ValueError(f"坐标 ({x}, {y}) 超出图片范围 ({width}, {height})")
    
    # 读取灰度图
    gray_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    # 获取BGR颜色值
    bgr_color = image[y, x]
    # 获取灰度值
    gray_value = gray_image[y, x]
    
    # 转换BGR为RGB
    rgb_color = tuple(reversed(bgr_color.tolist()))
    
    print(f"坐标 ({x}, {y}) 的颜色值:")
    print(f"BGR: {tuple(bgr_color)}")
    print(f"RGB: {rgb_color}")
    print(f"灰度值: {gray_value}")
    
    return bgr_color, gray_value

def draw_bbox(image_path, x1, y1, x2, y2, output_path=None):
    """
    在图片上绘制边界框并保存
    
    Args:
        image_path (str): 输入图片路径
        x1, y1 (int): 左上角坐标
        x2, y2 (int): 右下角坐标
        output_path (str): 输出图片路径，如果为None则自动生成
    """
    # 读取图片
    image = cv2.imread(image_path)
    
    # 检查图片是否成功读取
    if image is None:
        raise ValueError(f"无法读取图片: {image_path}")
    
    # 获取图片尺寸
    height, width = image.shape[:2]
    print(f"图片尺寸: 宽={width}, 高={height}")
    
    
    # 限制坐标在图片范围内
    x1 = max(0, min(x1, width-1))
    x2 = max(0, min(x2, width-1))
    y1 = max(0, min(y1, height-1))
    y2 = max(0, min(y2, height-1))
    
    print(f"调整后的坐标:")
    print(f"左上角: ({x1}, {y1})")
    print(f"右下角: ({x2}, {y2})")
    
    # 绘制边界框
    color = (0, 0, 255)  # 红色
    thickness = 1
    cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)
    
    # 如果没有指定输出路径，则自动生成
    if output_path is None:
        output_path = image_path.rsplit('.', 1)[0] + '_with_bbox.jpg'
    
    # 保存图片
    cv2.imwrite(output_path, image)
    print(f"已保存带边界框的图片到: {output_path}")
    
    # 输出边界框信息
    print(f"边界框坐标:")
    print(f"左上角: ({x1}, {y1})")
    print(f"右下角: ({x2}, {y2})")
    print(f"宽度: {x2 - x1}")
    print(f"高度: {y2 - y1}")

if __name__ == "__main__":
    # 使用示例
    image_path = "/home/chatmindai/project/zhangkun/Fast_Online_MTMCT/2. online_MTMC/preliminary/overlap_zones/c007_c006.png"  # 替换为你的图片路径
    x = 291  # 替换为你想要检查的x坐标
    y = 402  # 替换为你想要检查的y坐标
    
    try:
        # 获取颜色值
        bgr_color, gray_value = get_pixel_color(image_path, x, y)
    except Exception as e:
        print(f"错误: {e}")
    
    # 使用示例
    output_path = "./检查图片文件的情况的输出图片.jpg"  # 替换为你的图片路径
    # x1, y1 = 1446, 323  # 左上角坐标
    # x2, y2 = x1+531, y1+ 203 # 右下角坐标
    x1, y1 = 1529, 354  # 左上角坐标
    x2, y2 = x1+1896, y1+ 494 # 右下角坐标
    try:
        draw_bbox(image_path, x1, y1, x2, y2,output_path)
    except Exception as e:
        print(f"错误: {e}")