import os
import pandas as pd

"""
这段代码用来将生成的结果排序，排序顺序是 'Id', 'CameraId', 'FrameId' 然后写入到文件中
"""

def getData(fpath, sep='\s+|\t+|,'):
    names = ['CameraId', 'Id', 'FrameId', 'X', 'Y', 'Width', 'Height', 'Xworld', 'Yworld']
    try:
        df = pd.read_csv(
            fpath,
            sep=sep,
            index_col=None,
            skipinitialspace=True,
            header=None,
            names=names,
            engine='python'
        )
        return df
    except Exception as e:
        raise ValueError("Could not read input from %s. Error: %s" % (fpath, repr(e)))

def writeDataToTXT(df, filepath, delimiter='\t'):
    # 判断filepath的父目录存不存在
    if not os.path.exists(os.path.dirname(filepath)):
        os.makedirs(os.path.dirname(filepath))
    with open(filepath, 'w') as file:
        df.to_string(file, index=False, header=False)
        file.write('\n')

if __name__ == '__main__':
    detection_path = 'result/version/v2.txt'
    pred_data = getData(detection_path)
    sort_by = ['CameraId', 'FrameId' , 'Id']
    pred_data.sort_values(by=sort_by, inplace=True)
    writeDataToTXT(df=pred_data, filepath='debug_sort/results.txt')

    gt_path = 'test_gt.txt'
    gt_data = getData(gt_path)
    gt_data.sort_values(by=sort_by, inplace=True)
    writeDataToTXT(df=gt_data, filepath='debug_sort/gt.txt')
