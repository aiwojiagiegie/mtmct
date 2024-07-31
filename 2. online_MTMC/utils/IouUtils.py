def calculate_iou(bbox1, bbox2):
    """
    计算两个边界框之间的IOU (Intersection over Union).

    参数:
    bbox1: 第一个边界框，格式为 [left, top, width, height].
    bbox2: 第二个边界框，格式为 [left, top, width, height].

    返回:
    IOU值，范围在0到1之间.
    """
    # 将边界框的左上角和宽高转换为右下角坐标
    x1, y1, w1, h1 = bbox1[0], bbox1[1], bbox1[2], bbox1[3]
    x2, y2, w2, h2 = bbox2[0], bbox2[1], bbox2[2], bbox2[3]
    bbox1_right = x1 + w1
    bbox1_bottom = y1 + h1
    bbox2_right = x2 + w2
    bbox2_bottom = y2 + h2

    # 计算两个边界框的交集
    intersection_x_left = max(x1, x2)
    intersection_y_top = max(y1, y2)
    intersection_x_right = min(bbox1_right, bbox2_right)
    intersection_y_bottom = min(bbox1_bottom, bbox2_bottom)

    # 如果没有交集，IOU为0
    if intersection_x_right <= intersection_x_left or intersection_y_bottom <= intersection_y_top:
        return 0.0

    # 计算交集面积
    intersection_area = (intersection_x_right - intersection_x_left) * (intersection_y_bottom - intersection_y_top)

    # 计算并集面积
    bbox1_area = w1 * h1
    bbox2_area = w2 * h2
    union_area = bbox1_area + bbox2_area - intersection_area

    # 计算IOU
    iou = intersection_area / union_area
    return iou
if __name__ == '__main__':
    # 示例边界框
    bbox1 = [10, 10, 50, 50]  # 左上角在(10, 10)，宽50，高50
    bbox2 = [20, 20, 40, 40]  # 左上角在(20, 20)，宽40，高40
    bbox3 = [50, 50, 50, 50]  # 左上角在(50, 50)，宽50，高50
    bbox4 = [10, 10, 100, 100]  # 左上角在(10, 10)，宽100，高100

    # 测试函数
    print("IOU between bbox1 and bbox2:", calculate_iou(bbox1, bbox2))
    print("IOU between bbox1 and bbox3:", calculate_iou(bbox1, bbox3))
    print("IOU between bbox1 and bbox4:", calculate_iou(bbox1, bbox4))
    print("IOU between bbox2 and bbox3:", calculate_iou(bbox2, bbox3))
    print("IOU between bbox2 and bbox4:", calculate_iou(bbox2, bbox4))
    print("IOU between bbox3 and bbox4:", calculate_iou(bbox3, bbox4))

    # 测试完全不重叠的边界框
    bbox5 = [100, 100, 50, 50]
    print("IOU between bbox1 and bbox5:", calculate_iou(bbox1, bbox5))