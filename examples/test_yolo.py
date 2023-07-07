import sys
sys.path.append("..") 
import cv2 as cv
from robotpipe import YoloDetector
from robotpipe.utils import draw_labels

if __name__ == '__main__':
    # 构建对象
    yolo = YoloDetector()
    # 读取图片
    img = cv.imread("images/trump1.jpg")
    # 推理
    outputs = yolo.predict(img)
    # 绘制结果
    dst = yolo.draw_labels(img,outputs)
    # 显示结果
    cv.imshow("dst",dst)
    cv.waitKey()