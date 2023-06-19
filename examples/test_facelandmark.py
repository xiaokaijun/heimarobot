from heimarobot  import FaceLandmark
import cv2 as cv
import numpy as np
from heimarobot import draw_faces,draw_landmarks

if __name__ == '__main__':
    face_attr = FaceLandmark()
    trump1 = cv.imread('images/trump1.jpg')

    # 获取人脸特征106个特征点
    faces = face_attr.predict(trump1)

    retimg = draw_faces(trump1,faces)
    cv.imshow('trump1',retimg)

    retimg = draw_landmarks(trump1,faces)
    cv.imshow('trump1_landmark',retimg)
    cv.waitKey(0)

    

                    
