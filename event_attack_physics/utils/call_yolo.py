import os
import sys

# path1 = sys.path[0]
# print("path1",path1)
# os.chdir(path1)
import subprocess

def callback_yolox(yolox_path,input_img_name):
    yolo_checkpoint = "checkpoints/yolox_s.pth"
    #path = "/home/lgx/code/Attack_Event_Code/aaai_event_attack_human/attack_projects/YOLOX" #+" \n"
    #os.system(cmd)
    path = yolox_path
    cmd = f"conda activate rvt; cd {path};"
    cmd += f"python {path}/tools/demo.py image -n yolox-s -c {path}/{yolo_checkpoint} --path {input_img_name} --conf 0.25 --nms 0.45 --tsize 640 --save_result --device gpu"
    # 使用subprocess.run()运行多句命令，并获取输出
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)

import os
import re
import cv2
import numpy as np

def find_pattern(text,pattern):
    
    match = re.search(pattern, text)

    if match:
        # 提取括号内的内容
        numbers_str = match.group(1)
        # 使用正则表达式提取所有的浮点数
        numbers = re.findall(r'\d+\.\d+', numbers_str)
        # 将字符串转换为浮点数
        float_numbers = [float(num) for num in numbers]
        #print("提取的浮点数：", float_numbers)
        return float_numbers
    else:
        #print("未找到匹配的结果")
        return None
        
def find_pattern_bbox(str1,pattern_list):
    #box_pattern = r"\[\[.*?\]\]"
    extracted_array = None
    index = 0
    for pattern in pattern_list:
        index +=1 
        print("pattern",pattern)
        matches = re.findall(pattern, str1)
        print(matches)
        if matches:
            if index == 1:
                numbers_str = matches[0].replace("[[","").replace("]]","")
            elif index ==2:
                numbers_str = matches[0].replace("[[","").replace("]\n","")
            print("numbers_str",numbers_str)
            numbers = np.fromstring(numbers_str, sep=' ').astype('int')
            # 将一维数组重新整理为二维数组
            extracted_array = numbers.reshape(-1, 4)
            print("Extracted NumPy array:")
            print(extracted_array)
            break
    if extracted_array is None:    
        return None
        #extracted_array = None

    return extracted_array


def callback_onnx_yolox_s(yolox_path, input_img_name,output_dir,cuda_id):
   # yolo_checkpoint = os.path.join(yolox_path,"checkpoints/yolox_s.pth")
    
    #path = "/home/lgx/code/Attack_Event_Code/aaai_event_attack_human/attack_projects/YOLOX" #+" \n"
    #os.system(cmd)
    path = yolox_path
    cmd = f" cd {path};" #conda activate rvt;
    #shape = cv2.
    image = cv2.imread(input_img_name)
    shape = image.shape
    height = shape[0]
    width = shape[1]
    #print(shape)
    
    #cmd += f"python {path}/demo/onnx_inference.py image -n yolox-s -c {path}/{yolo_checkpoint} --path {input_img_name} --conf 0.25 --nms 0.45 --tsize 640 --save_result --device gpu"
    # 使用subprocess.run()运行多句命令，并获取输出
    onnx_path = 'checkpoints/yolox_s.onnx'
    cmd += f"CUDA_VISIBLE_DEVICES={cuda_id} python demo/ONNXRuntime/onnx_inference.py -m {path}/{onnx_path} -i {input_img_name} -o {output_dir} -s 0.3 --input_shape {height},{width}"
    print("cmd",cmd)
    #os.system(cmd)
    #print("")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    print("result",result)
    # 输出命令行的标准输出
    #print(result.stdout)

    # 输出命令行的标准错误
    #print(result.stderr)

    # 输出命令的返回码
    #print("返回码:", result.returncode)

    return result.stdout

def callback_yolov7(yolov7_path, input_img_name,output_dir,cuda_id):
   # yolo_checkpoint = os.path.join(yolox_path,"checkpoints/yolox_s.pth")
    
    #path = "/home/lgx/code/Attack_Event_Code/aaai_event_attack_human/attack_projects/YOLOX" #+" \n"
    #os.system(cmd)
    path = yolov7_path
    cmd = f" cd {path};" #conda activate rvt;
    #shape = cv2.
    #image = cv2.imread(input_img_name)

    
    cmd += f"CUDA_VISIBLE_DEVICES={cuda_id} python detect.py --weights yolov7.pt --conf 0.25 --img-size 640 --source {input_img_name} --save-dir {output_dir}"
    
    print("cmd",cmd)
    #os.system(cmd)
    #print("")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    print("result",result)


    return result.stdout


# def find_pattern_score(str1,pattern):
#     #box_pattern = r"\[\[.*?\]\]"
#     matches = re.findall(pattern, str1)
#     print(matches)
#     if matches:
#         numbers_str = matches[0]
#         numbers_str = numbers_str.replace("[","").replace("]","").split(' ')[1]#[-1]
#         #print("numbers_str",numbers_str)

#         score = float(numbers_str)#np.fromstring(numbers_str, sep=' ').astype('float')
#         #print("score")
#         print(score)
#     else:
#         print("No match found.")
        
#     return score


def get_subdirectories(path):
    subdirectories = [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]
    return subdirectories

if __name__ == "__main__":
    target_path = "/home/lgx/data/event_attack_data/GEN1_Processed/test"  # 替换为你要查找的路径
    subdirs = get_subdirectories(target_path)
    print(subdirs)

