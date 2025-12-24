from PIL import Image, ImageDraw, ImageFilter

# Load the images
uv_texture_path = "/home/lgx/code/AAAI25/Attack/submission/event_20/output_data/imm/uv_texture.png"  # 替换为你的UV纹理图像路径
vis_mask_path = "/home/lgx/code/AAAI25/Attack/submission/event_20/output_data/imm/vis_mask.png"      # 替换为你的轮廓图像路径


uv_texture = Image.open(uv_texture_path)
vis_mask = Image.open(vis_mask_path)

# # Resize vis_mask to the same size as uv_texture
# vis_mask = vis_mask.resize(uv_texture.size, Image.NEAREST)

# # Convert vis_mask to grayscale to ensure binary mask (black and white only)
# vis_mask = vis_mask.convert("L")

# # Create a new image to draw the outline on uv_texture
# result_image = uv_texture.copy()
# draw = ImageDraw.Draw(result_image)

# # Threshold to consider anything darker than this as black
# threshold = 128

# # Create a blurred version of the mask to detect edges
# blurred_mask = vis_mask.filter(ImageFilter.GaussianBlur(radius=2))

# # Find edges by checking the difference between the original mask and blurred mask
# for x in range(1, vis_mask.width - 1):
#     for y in range(1, vis_mask.height - 1):
#         original_pixel = vis_mask.getpixel((x, y))
#         blurred_pixel = blurred_mask.getpixel((x, y))
        
#         # If there's a significant difference, it means it's an edge
#         if abs(original_pixel - blurred_pixel) > threshold:
#             draw.point((x, y), fill="black")

# # Save the result image
# result_image_path = "/mnt/data/uv_texture_with_outline_only.png"
# result_image.save(result_image_path)






uv_texture = Image.open(uv_texture_path)
vis_mask = Image.open(vis_mask_path)

# Resize vis_mask to the same size as uv_texture
vis_mask = vis_mask.resize(uv_texture.size, Image.NEAREST)

# Convert vis_mask to grayscale to ensure binary mask (black and white only)
vis_mask = vis_mask.convert("L")

# Create a new image to draw the outline on uv_texture
result_image = uv_texture.copy()
draw = ImageDraw.Draw(result_image)
threshold = 128
# Threshold to consider anything darker than this as black
offset = 2
for x in range(offset, vis_mask.width - offset):
    for y in range(offset, vis_mask.height - offset):
        original_pixel = vis_mask.getpixel((x, y))
        neighbor_pixel_1 = vis_mask.getpixel((x + offset, y))
        neighbor_pixel_2 = vis_mask.getpixel((x, y + offset))
        
        # If there's a difference, it means it's an edge
        if abs(original_pixel - neighbor_pixel_1) > threshold or abs(original_pixel - neighbor_pixel_2) > threshold:
            draw.point((x, y), fill="black")



# Save the result image
result_image_path = "/home/lgx/code/AAAI25/Attack/submission/event_20/output_data/imm/uv_texture_with_outline.png"
result_image.save(result_image_path)

# Show the result (optional)

