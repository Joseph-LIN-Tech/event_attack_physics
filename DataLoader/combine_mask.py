import cv2
import numpy as np
import os

# 文件夹路径
folder_path = '/home/lgx/code/NMI/attack_event_NMI_mask_full_mask_64cellsize/utils/sam_mask'

# 读取所有mask图片
mask_files = [
    #'mask_finger_.png',
   # 'mask_finger.png',
    #'mask_foot.png',
    'mask_hand_.png',
    'mask_hand.png',
    # 'mask_leg_.png',
    # 'mask_leg.png'
]

# 初始化空的合并mask
combined_mask = None

# 迭代每个mask文件并合并
for mask_file in mask_files:
    mask_path = os.path.join(folder_path, mask_file)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    
    if combined_mask is None:
        combined_mask = np.zeros_like(mask)
    
    combined_mask = cv2.bitwise_or(combined_mask, mask)

# 保存合并后的mask图片
output_path = os.path.join(folder_path, 'combined_mask_hand.png')
cv2.imwrite(output_path, combined_mask)

print(f'合并后的mask已保存到: {output_path}')
