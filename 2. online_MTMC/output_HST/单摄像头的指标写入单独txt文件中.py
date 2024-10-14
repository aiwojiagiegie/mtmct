from outputs.eval import *


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


def my_print_result(summary,camera_id):
    dict_list = summary.to_dict(orient='records')
    format_str = "{:<20} {:<12} {}"
    float_format = "{:<20} {:<12.6%} {}"
    print(f'==================camera_id={camera_id}分析结果======================',file=f)
    print(format_str.format("指标", "当前结果", "指标解释"),file=f)
    for record in dict_list:
        for key, value in record.items():
            # 计算差值和符号
            if isinstance(value, float):
                if value.is_integer():
                    print(format_str.format(key,  int(value), info[key]),file=f)
                else:
                    print(float_format.format(key, value,  info[key]),file=f)
            else:
                print(format_str.format(key,  value,  info[key]),file=f)


def write_results(gt, pred, cam_id, mread=False, dstype='validation'):
    # 将文件写入到 ./debug_单摄像头分析/cam_id_test.txt中
    # 取消对头的输出
    gt.to_csv(f'./debug_单摄像头分析/{cam_id}_gt.txt', index=False, header=False)
    pred.to_csv(f'./debug_单摄像头分析/{cam_id}_pred.txt', index=False, header=False)


if __name__ == '__main__':
    version = 'v3'
    pred_path = f'/home/chatmindai/project/zhangkun/Fast_Online_MTMCT/2. online_MTMC/output_HST/result/version/{version}.txt'
    gt_path = '/home/chatmindai/project/zhangkun/Fast_Online_MTMCT/2. online_MTMC/output_HST/test_gt.txt'

    gt = readData(gt_path)
    unique_camera_ids = gt['CameraId'].unique()
    test_grouped_dataframes = {}
    for camera_id in unique_camera_ids:
        test_grouped_dataframes[camera_id] = gt[gt['CameraId'] == camera_id]

    pred = readData(pred_path)
    unique_camera_ids = pred['CameraId'].unique()
    pred_grouped_dataframes = {}
    for camera_id in unique_camera_ids:
        pred_grouped_dataframes[camera_id] = pred[pred['CameraId'] == camera_id]

    for camera_id in unique_camera_ids:
        write_results(test_grouped_dataframes[camera_id], pred_grouped_dataframes[camera_id],camera_id)
