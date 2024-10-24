import cv2
from yolov10.ultralytics import YOLOv10
import os

def detect_and_draw(image_path, output_path, model_path):
    # 加载YOLOv10模型
    model = YOLOv10(model_path)
    
    # 读取图片
    image = cv2.imread(image_path)
    if image is None:
        print(f"无法读取图片: {image_path}")
        return
    
    # 进行目标检测
    results = model(image)[0]
    
    # 获取检测结果
    boxes = results.boxes.data.tolist()
    names = results.names
    
    # 在图片上绘制边界框和标签
    for box in boxes:
        x1, y1, x2, y2, conf, class_id = box
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
        
        # 绘制边界框
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # 添加标签
        label = f"{names[int(class_id)]}: {conf:.2f}"
        cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # 保存结果图片
    cv2.imwrite(output_path, image)
    print(f"已保存检测结果到: {output_path}")

if __name__ == "__main__":
    model_path = '/home/chatmindai/project/zhangkun/yolov10/runs/detect/UA-DETRAC_pre/model_name_yolov10s.pt/epochs_200/batch_322/weights/best.pt'
    image_path = "/home/chatmindai/project/zhangkun/Fast_Online_MTMCT/2. online_MTMC/output_HST/逐帧目标检测/43/MTMCT中对图片进行debug/43_f0007.jpg"
    output_path = "./image_detected.jpg"
    
    detect_and_draw(image_path, output_path, model_path)
