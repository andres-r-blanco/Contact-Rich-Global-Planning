import os
import numpy as np
import h5py
import pandas

# 获取所有npz文件
files_all = [f for f in os.listdir("/home/angchen/xac/data/world_model/test_data_100") if f.endswith(".npz")]
# files_all = ['/home/angchen/xac/data/world_model/test_data/ee_trajectory20240818-195143.npz']
file_num = 2
save_dir = f"/home/angchen/xac/data/world_model/test_data_h5_{file_num}"

# 确保保存目录存在
os.makedirs(save_dir, exist_ok=True)

# 初始化计数器
count = 0
file_index = 0

# 分批处理npz文件
for i in range(0, len(files_all), file_num):
    # 创建新的h5文件
    h5_file_path = os.path.join(save_dir, f'test_data_{file_index}.h5')
    h5_file = h5py.File(h5_file_path, 'w')
    file_index += 1

    # 处理当前批次的npz文件
    for file in files_all[i:i+file_num]:
        count += 1
        print(f"Loading data from {file}")
        with np.load(os.path.join("/home/angchen/xac/data/world_model/test_data_100", file), allow_pickle=True) as data:
            data = data["arr_0"]
            fname = file.split('.')[0]
            grp = h5_file.create_group(str(count))

            df_frame = pandas.DataFrame(columns=data[0].keys(), index=None)
            data_dict = {key: [] for key in data[0].keys()}
            for j in range(len(data)):
                for key in data[j].keys():
                    data_dict[key].append(np.array(data[j][key]))
            for key in data_dict.keys():
                data_dict[key] = np.array(data_dict[key])
            # 保存数据到h5文件
            for key in data_dict.keys():
                grp.create_dataset(key, data=data_dict[key])

    # 关闭当前h5文件
    h5_file.close()