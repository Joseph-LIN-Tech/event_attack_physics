import h5py

# 打开H5文件
h5file = "/home/lgx/code/ECCV2024/white_box_attack/LETGAN/Results/output.h5"
with h5py.File(h5file, 'r') as file:
    # 列出文件中所有的主键
    print("Keys: %s" % file.keys())
    a_group_key = list(file.keys())[0]
    
    # 获取其中一个数据集
    data = list(file[a_group_key])
    
    # 打印数据
    print(data)
    print(data['ps'])
    print(data['ts'])
    print(data['xs'])
