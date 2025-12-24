import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
import os
# 读取图像
image_path = '/home/lgx/code/NMI/attack_event_NMI_loss_Patch_wo_uv_init_sam/Data/smpl_uv.png'
image = cv2.imread(image_path)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# 初始化 SAM 模型
sam_checkpoint = "/home/lgx/code/NMI/attack_event_NMI_loss_Patch_wo_uv_init_sam/utils/sam_vit_h_4b8939.pth"  # 下载的SAM模型的路径
device = "cuda" if torch.cuda.is_available() else "cpu"
model_type = "vit_h"
sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)

# 使用 SAM 自动生成掩码
#mask_generator = SamAutomaticMaskGenerator(sam)


# 配置 SAM 自动掩码生成器为 "everything" 模式
mask_generator = SamAutomaticMaskGenerator(
    model=sam,
    points_per_side=32,  # 增加点的数量以覆盖所有区域
    pred_iou_thresh=0.9,  # 预测IoU阈值，可以根据需要调整
    stability_score_thresh=0.9,  # 稳定性评分阈值
    crop_n_layers=1,  # 裁剪层数
    crop_n_points_downscale_factor=2,  # 裁剪点缩放因子
    min_mask_region_area=100,  # 掩码最小区域大小
)

masks = mask_generator.generate(image_rgb)

# 创建保存掩码的文件夹
mask_folder = '/home/lgx/code/NMI/attack_event_NMI_loss_Patch_wo_uv_init_sam/utils/masks'
os.makedirs(mask_folder, exist_ok=True)

print("len(masks)",len(masks))
# 可视化和保存掩码
fig, axes = plt.subplots(1, len(masks), figsize=(20, 20))
for i, mask in enumerate(masks):
    mask_image = mask['segmentation'].astype(np.uint8) * 255
    mask_filename = f'{mask_folder}/mask_{i}.png'
    cv2.imwrite(mask_filename, mask_image)
    if i < len(axes):
        axes[i].imshow(mask_image, cmap='gray')
        axes[i].axis('off')

plt.show()