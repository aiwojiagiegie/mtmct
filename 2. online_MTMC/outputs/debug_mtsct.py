from eval import *


def eval(test, pred, **kwargs):
    if test is None:
        return None
    mread = kwargs.pop('mread', False)
    dstype = kwargs.pop('dstype', 'train')
    roidir = kwargs.pop('roidir', 'ROIs')

    mh = mm.metrics.create()
    pred = removeOutliersROI(pred, dstype=dstype, roidir=roidir)
    pred = removeRepetition(pred)
    return compare_dataframes_mtmc(test, pred, mh)


baseline = {
    6: {'idf1': 0.7960704607046071, 'idp': 0.6986769733908131, 'idr': 0.9250147608738437,
          'recall': 0.9348553434363315, 'precision': 0.7061097071502899, 'num_unique_objects': 124,
          'mostly_tracked': 111, 'partially_tracked': 12, 'mostly_lost': 1, 'num_false_positives': 1977,
          'num_misses': 331, 'num_switches': 2, 'num_fragmentations': 41, 'mota': 0.5453650856130683,
          'motp': 0.2550539289772449, 'num_transfer': 3, 'num_ascend': 2, 'num_migrate': 3, 'num_frames': 2033,
          'idfp': 2027.0, 'idfn': 381.0, 'idtp': 4700.0},
    7: {'idf1': 0.9249166379211222, 'idp': 0.8975675072528454, 'idr': 0.9539848197343453,
          'recall': 0.9705882352941176, 'precision': 0.913189020307967, 'num_unique_objects': 90, 'mostly_tracked': 86,
          'partially_tracked': 2, 'mostly_lost': 2, 'num_false_positives': 389, 'num_misses': 124, 'num_switches': 3,
          'num_fragmentations': 11, 'mota': 0.8776091081593927, 'motp': 0.2508358658493914, 'num_transfer': 3,
          'num_ascend': 3, 'num_migrate': 3, 'num_frames': 1875, 'idfp': 459.0, 'idfn': 194.0, 'idtp': 4022.0},
    8: {'idf1': 0.8018549747048904, 'idp': 0.7729612571118938, 'idr': 0.832992700729927, 'recall': 0.9483211678832116,
          'precision': 0.8799783256570035, 'num_unique_objects': 99, 'mostly_tracked': 81, 'partially_tracked': 4,
          'mostly_lost': 14, 'num_false_positives': 443, 'num_misses': 177, 'num_switches': 6, 'num_fragmentations': 9,
          'mota': 0.8172262773722627, 'motp': 0.21649710289717095, 'num_transfer': 1, 'num_ascend': 6, 'num_migrate': 1,
          'num_frames': 1804, 'idfp': 838.0, 'idfn': 572.0, 'idtp': 2853.0},
    9: {'idf1': 0.9106482839542388, 'idp': 0.8808307797071842, 'idr': 0.9425552586835074,
          'recall': 0.9486276414865193, 'precision': 0.8865055044830326, 'num_unique_objects': 137,
          'mostly_tracked': 118, 'partially_tracked': 18, 'mostly_lost': 1, 'num_false_positives': 1000,
          'num_misses': 423, 'num_switches': 3, 'num_fragmentations': 55, 'mota': 0.8268156424581006,
          'motp': 0.26723392249458666, 'num_transfer': 3, 'num_ascend': 1, 'num_migrate': 2, 'num_frames': 2110,
          'idfp': 1050.0, 'idfn': 473.0, 'idtp': 7761.0}
}


def my_print_result(summary,camera_id):
    dict_list = summary.to_dict(orient='records')
    format_str = "{:<20} {:<15} {:<12} {:10} {:4} {}"
    float_format = "{:<20} {:<15.6%} {:<12.6%} {:10.6%} {:4} {}"
    print(f'==================camera_id={camera_id}分析结果======================',file=f)
    print(format_str.format("指标", "baseline", "当前结果", "差值", "好坏", "指标解释"),file=f)
    for record in dict_list:
        for key, value in record.items():
            # 计算差值和符号
            difference = value - baseline[camera_id][key]
            symbol = "↑" if difference > 0 else "↓" if difference < 0 else ""
            if isinstance(value, float):
                if value.is_integer():
                    print(format_str.format(key, int(baseline[camera_id][key]), int(value), difference, symbol, info[key]),file=f)
                else:
                    print(float_format.format(key,baseline[camera_id][key], value, difference, symbol, info[key]),file=f)
            else:
                print(format_str.format(key, baseline[camera_id][key], value, difference, symbol, info[key]),file=f)


def calculate_results(test, pred,cam_id, mread=False, dstype='validation',
                      roidir='/data2/zhangkun/project/Fast_Online_MTMCT/dataset/AIC19/validation/S02'):
    try:
        summary = eval(test, pred, mread=mread, dstype=dstype, roidir=roidir)
        # print(summary.to_dict(orient='records'))
        my_print_result(summary,cam_id)
    except Exception as e:
        if mread:
            print('{"error": "%s"}' % repr(e))
        else:
            print("Error: %s" % repr(e))
        traceback.print_exc()


if __name__ == '__main__':
    test_path, pred_path = 'ground_truth_validation.txt', f'result/v5.txt'

    test = readData(test_path)
    unique_camera_ids = test['CameraId'].unique()
    test_grouped_dataframes = {}
    for camera_id in unique_camera_ids:
        test_grouped_dataframes[camera_id] = test[test['CameraId'] == camera_id]

    pred = readData(pred_path)
    unique_camera_ids = pred['CameraId'].unique()
    pred_grouped_dataframes = {}
    for camera_id in unique_camera_ids:
        pred_grouped_dataframes[camera_id] = pred[pred['CameraId'] == camera_id]
    with open('单摄像头跟踪结果分析.txt', 'w') as f:
        for camera_id in unique_camera_ids:
            calculate_results(test_grouped_dataframes[camera_id], pred_grouped_dataframes[camera_id],camera_id)
