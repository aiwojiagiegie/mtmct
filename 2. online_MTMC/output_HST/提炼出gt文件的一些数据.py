def get_bbox_data(txt_path):
    bbox_data = {}
    min_area = float('inf')  # 初始化最小面积为无穷大
    
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
            
            # 计算并更新最小面积
            area = width * height
            min_area = min(min_area, area)
            
            if camera_id not in bbox_data:
                bbox_data[camera_id] = {}
            if frame_number not in bbox_data[camera_id]:
                bbox_data[camera_id][frame_number] = []
            bbox_data[camera_id][frame_number].append((left, top, width, height, car_id))
    
    return bbox_data, min_area

if __name__ == '__main__':
    txt_path = 'test_gt.txt'
    bbox_data, min_area = get_bbox_data(txt_path)
    print(f"最小的边界框面积是: {min_area}")

