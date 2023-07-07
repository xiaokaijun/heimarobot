import onnxruntime as rt
from PIL import Image
import numpy as np
onnx_path = r"C:/Users/KAI/.robotpipe/models/scrfd_person_2.5g.onnx"
session = rt.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])

img = Image.open("images/trump1.jpg")
print(img)

#  对图像进行处理 --->BGR
img = np.array(img)[:, :, [2,1,0]].astype(np.float32)

# HWC --> CHW
img = np.transpose(img,[2,0,1])

# 对数据进行归一化
mean_vec = np.array([102.9801,115.9465,122.7717])
for i in range(img.shape[0]):
    img[i,:,:] = img[i,:,:] - mean_vec[i]

# pad to be divisible of32
import math
padded_h = int(math.ceil(img.shape[1]/32)*32)
padded_w = int(math.ceil(img.shape[2]/32)*32)

padded_img = np.zeros((3,padded_h,padded_w),dtype=np.float32)
padded_img[:,:img.shape[1],:img.shape[2]]=img

img = padded_img


input_name = session.get_inputs()[0].name
print("Input name  :", input_name)
input_shape = session.get_inputs()[0].shape
print("Input shape :", input_shape)
input_type = session.get_inputs()[0].type
print("Input type  :", input_type)

result = session.run(None,{input_name:[img]})

print(result)