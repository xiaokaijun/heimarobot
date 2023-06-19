from face_recognition import FaceRecognition
from face_attribute import FaceAttribute
import cv2 as cv
import numpy as np

if __name__ == '__main__':
    face_attr = FaceAttribute()
    trump1 = cv.imread('images/trump1.jpg')
    trump1 = cv.imread('images/1.jpg')
    
    
    # 获取人脸特征
    faces = face_attr.predict(trump1)
    # 1 表示男性， 0，表示女性
    print(faces[0])

    

                    
