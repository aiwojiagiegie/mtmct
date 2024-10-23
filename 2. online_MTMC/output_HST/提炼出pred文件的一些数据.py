# 读取pred文件，提取出track_id为20的数据，然后打印每个摄像头中的起止帧的情况

import pandas as pd
from prettytable import PrettyTable

def read_pred_file(file_path):
    df = pd.read_csv(file_path, sep=' ', header=None)
    df.columns = ['camera_id', 'track_id', 'frame_id', 'x', 'y', 'w', 'h', 'conf', 'class']
    return df

def get_track_info(df, id_list):
    results = []
    for track_id in id_list:
        df_id = df[df['track_id'] == track_id]
        for camera_id in df_id['camera_id'].unique():
            df_camera = df_id[df_id['camera_id'] == camera_id]
            results.append([
                track_id,
                camera_id,
                df_camera["frame_id"].min(),
                df_camera["frame_id"].max()
            ])
    return sorted(results, key=lambda x: (x[0], x[1]))

def create_pretty_table(results):
    table = PrettyTable()
    table.field_names = ["Track ID", "摄像头ID", "起始帧", "结束帧"]
    for row in results:
        table.add_row(row)
    return table

def main():
    pred_path = '/home/chatmindai/project/zhangkun/Fast_Online_MTMCT/2. online_MTMC/output_HST/result/version/v17.txt'
    id_list = [4, 1, 0]  # 可以根据需要修改这个列表

    df = read_pred_file(pred_path)
    results = get_track_info(df, id_list)
    table = create_pretty_table(results)
    print(table)

if __name__ == "__main__":
    main()
