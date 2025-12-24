import os
import glob

def delete_shape_npz_files(folder_path):
    # 使用 glob 模块找到所有名为 shape.npz 的文件
    file_pattern = os.path.join(folder_path, '**', 'shape.npz')
    files_to_delete = glob.glob(file_pattern, recursive=True)

    # 遍历找到的文件并删除
    for file_path in files_to_delete:
        try:
            os.remove(file_path)
            print(f"Deleted: {file_path}")
        except Exception as e:
            print(f"Error deleting {file_path}: {e}")

# 示例用法
folder_path = '/home/lgx/data/whitebox_attack_cmupose'  # 替换为实际的文件夹路径
delete_shape_npz_files(folder_path)
