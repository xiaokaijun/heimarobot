from face_swapper import FaceSwapper

import cv2 as cv
import numpy as np

if __name__ == '__main__':
    face_swapper = FaceSwapper()
    trump1 = cv.imread('images/zp.jpg')
    target = cv.imread('images/2.jpg')

    sourceFace = face_swapper.get_source_face(trump1)

    dst = face_swapper.predict(sourceFace, target)
   
    cv.imshow("dst", dst)
    cv.waitKey()

    

                    
