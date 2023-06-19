from heimarobot import FaceRecognition
import cv2 as cv
import numpy as np

if __name__ == '__main__':
    face_recognition = FaceRecognition()
    trump1 = cv.imread('images/1.jpg')
    trump2 = cv.imread('images/trump2.jpg')
    
    # 获取人脸特征
    feats1 = face_recognition.predict(trump1)
    feats2 = face_recognition.predict(trump2)
    
    sim = face_recognition.compute_sim(feats1,feats2)
    print(feats1)
    # 根据相似度,输出结果
    if sim<0.2:
        conclu = 'they are not the same'
    elif sim>=0.2 and sim<0.28:
        conclu = 'they are looks like the same people'
    else:
        conclu = 'they are the same people'

    print(sim,conclu)

    

                    
