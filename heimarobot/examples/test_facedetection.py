from face_detection import FaceDetection
import cv2 as cv
import numpy as np

if __name__ == '__main__':
    face_detection = FaceDetection()
    img = cv.imread('images/1.jpg')
    bboxes,kpss = face_detection.predict(img)

    bboxes = np.intp(bboxes)
   
    for bbox in bboxes:
        x1,y1,x2,y2 = bbox[0:4]
        cv.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)
    
    cv.imshow('img',img)
    cv.waitKey(0)

                    