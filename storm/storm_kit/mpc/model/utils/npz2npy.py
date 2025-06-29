import numpy as np
import os

def convert_npz_to_npy(npz_folder, output_folder):
    # 确保输出目录存在
    os.makedirs(output_folder, exist_ok=True)
    
    # 遍历 npz 文件夹中的所有 .npz 文件
    for file_name in os.listdir(npz_folder):
        if file_name.endswith('.npz'):
            npz_file_path = os.path.join(npz_folder, file_name)
            # 加载 .npz 文件
            npz_data = np.load(npz_file_path, allow_pickle=True)
            
            # 由于每个 .npz 文件中只有一个数组，直接取第一个数组
            array_name = npz_data.files[0]  # 获取数组的名字
            array_data = npz_data[array_name]
            
            # 构建 .npy 文件路径
            base_name = os.path.splitext(file_name)[0]
            npy_file_path = os.path.join(output_folder, f"{base_name}.npy")
            
            # 保存为 .npy 文件
            np.save(npy_file_path, array_data)
            print(f"Saved {npy_file_path}")

# 使用示例
npz_folder_path = '/home/angchen/xac/data/world_model/test_data'
output_folder_path = "/home/angchen/xac/data/world_model/test_data_npy"
convert_npz_to_npy(npz_folder_path, output_folder_path)
