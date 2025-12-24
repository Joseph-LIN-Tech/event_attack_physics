# Calculate the scale factor to resize the image for 1m x 1m printing without losing aspect ratio

from PIL import Image, ImageDraw, ImageFilter
from PIL import Image

from PIL import Image

from PIL import Image

def split_image_to_a4(image_path, output_path, output_size_mm=1000, dpi=300):
    # Load the image
    # Load the original image
    image = Image.open(image_path)
    
    # Original image size in pixels
    original_size_px = image.size  # (width, height)
    
    # Target physical size is 1m x 1m
    target_size_mm = (output_size_mm, output_size_mm)  # 1 meter x 1 meter in millimeters
    
    # Convert target size from millimeters to inches (1 inch = 25.4 mm)
    target_size_in = (target_size_mm[0] / 25.4, target_size_mm[1] / 25.4)
    
    # Calculate the target pixel size based on the desired DPI
    target_size_px = (int(target_size_in[0] * dpi), int(target_size_in[1] * dpi))
    
    # Resize the original image to the target size
    resized_image = image.resize(target_size_px, Image.LANCZOS)
    
    # A4 paper size in pixels at the given DPI
    a4_width_px = int(210 * dpi / 25.4)  # 210mm is A4 width
    a4_height_px = int(297 * dpi / 25.4)  # 297mm is A4 height
    
    # Number of A4 pages needed to cover the entire image
    pages_x = (target_size_px[0] + a4_width_px - 1) // a4_width_px
    pages_y = (target_size_px[1] + a4_height_px - 1) // a4_height_px
    
    # Split the image into A4-sized chunks
    output_images = []
    for j in range(pages_y):
        for i in range(pages_x):
            left = i * a4_width_px
            upper = j * a4_height_px
            right = min(left + a4_width_px, target_size_px[0])
            lower = min(upper + a4_height_px, target_size_px[1])
            
            # Crop the image to get the current A4 page
            a4_image = Image.new('RGB', (a4_width_px, a4_height_px), (255, 255, 255))
            crop_box = resized_image.crop((left, upper, right, lower))
            a4_image.paste(crop_box, (0, 0))
            output_images.append(a4_image)
    
    # Save each chunk as a separate image file
    output_paths = []
    for idx, a4_image in enumerate(output_images):
        output_path_ = f"{output_path}/a4_page_{idx + 1}.png"
        a4_image.save(output_path_)
        output_paths.append(output_path_)
    
    return output_paths


# Load the image
image_path = "/home/lgx/code/AAAI25/Attack/submission/event_20/output_data/imm/uv_texture_with_outline.png"  # 替换为你的图像路径
output_path = "/home/lgx/code/AAAI25/Attack/submission/event_20/output_data/printable_a4"
n = 1  # Adjust this value to change the size of the print (n meters by n meters)
output_paths = split_image_to_a4(image_path, output_path, output_size_mm=500)

# Output paths of the generated A4 pages
output_paths

