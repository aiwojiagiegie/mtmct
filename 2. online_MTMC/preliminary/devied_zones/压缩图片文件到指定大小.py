import os
from PIL import Image
import sys
from pathlib import Path

def resize_image(input_path, output_path, target_width=1920, target_height=1080):
    """调整图片尺寸到1920x1080，保持比例并填充黑色背景"""
    try:
        # 打开图片
        img = Image.open(input_path)
        
        # 如果是RGBA模式，转换为RGB
        if img.mode == 'RGBA':
            img = img.convert('RGB')
        
        # 计算调整后的尺寸，保持原始比例
        width, height = img.size
        ratio = min(target_width/width, target_height/height)
        new_width = int(width * ratio)
        new_height = int(height * ratio)
        
        # 调整图片大小
        img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        # 创建黑色背景
        background = Image.new('RGB', (target_width, target_height), 'black')
        
        # 将调整后的图片粘贴到背景中央
        offset_x = (target_width - new_width) // 2
        offset_y = (target_height - new_height) // 2
        background.paste(img, (offset_x, offset_y))
        
        # 创建输出目录
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # 保存图片
        background.save(output_path, 'JPEG', quality=95)
        return True
        
    except Exception as e:
        print(f"处理图片 {input_path} 时出错: {str(e)}")
        return False

def process_directory(input_dir, output_dir):
    """递归处理目录下的所有图片文件"""
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    
    # 支持的图片格式
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
    
    # 统计信息
    total_files = 0
    processed_files = 0
    
    # 遍历所有文件
    for input_path in input_dir.rglob('*'):
        if input_path.suffix.lower() in image_extensions:
            total_files += 1
            
            # 计算输出路径，保持相对路径结构
            relative_path = input_path.relative_to(input_dir)
            output_path = output_dir / relative_path
            
            # 确保输出文件扩展名为jpg
            output_path = output_path.with_suffix('.jpg')
            
            print(f"处理: {input_path}")
            if resize_image(str(input_path), str(output_path)):
                processed_files += 1
    
    return total_files, processed_files

def main():
    
    input_dir = 'AIC19'
    output_dir = 'AIC19_compressed'
    
    if not os.path.exists(input_dir):
        print(f"输入目录 {input_dir} 不存在!")
        sys.exit(1)
    
    print(f"开始处理图片...")
    print(f"输入目录: {input_dir}")
    print(f"输出目录: {output_dir}")
    print(f"目标尺寸: 1920x1080")
    
    total, processed = process_directory(input_dir, output_dir)
    
    print(f"\n处理完成!")
    print(f"总文件数: {total}")
    print(f"成功处理: {processed}")
    print(f"失败数量: {total - processed}")

if __name__ == "__main__":
    main()
