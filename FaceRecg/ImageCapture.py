import cv2


imgSaveFolder = r'D:\hewei\program\python\FaceRecgData\myImage'
imageCount = 0

#调用笔记本内置摄像头，所以参数为0，如果有其他的摄像头可以调整参数为1，2
cap=cv2.VideoCapture(0)

while True:
    imageCount +=1
    imgSaveFilename = imgSaveFolder + '\\' + str(imageCount) + '.jpg'

    #从摄像头读取图片
    sucess,img=cap.read()

    #显示摄像头，背景是灰度。
    cv2.imwrite(imgSaveFilename,img)

    #保持画面的持续。
    k=cv2.waitKey(50)
    if imageCount == 100:
        #通过esc键退出摄像
        cv2.destroyAllWindows()
        break

#关闭摄像头
cap.release()