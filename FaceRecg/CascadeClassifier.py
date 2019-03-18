# '''
#     __author__='HW'
#     __data__='201903'
#     __desc__='人脸识别'
# '''

import cv2

imagePath = r'D:\hewei\program\python\FaceRecgData\lfw\Aaron_Eckhart\Aaron_Eckhart_0001.jpg'
classifierPath = r'C:\Users\lanliying\AppData\Local\Programs\Python\Python36\Lib\site-packages' \
                 r'\cv2\data\haarcascade_frontalface_default.xml'


# 获取训练好的人脸的参数数据
# face_cascade = cv2.CascadeClassifier(classifierPath)
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
face_cascade.load(classifierPath)

image = cv2.imread(imagePath)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 探测图片中的人脸

faces = face_cascade.detectMultiScale(

    gray,

    scaleFactor=1.15,

    minNeighbors=5,

    minSize=(5, 5),
)

print("发现{0}个人脸!".format(len(faces)))

for (x, y, w, h) in faces:
    cv2.rectangle(image,(x,y),(x+w,y+w),(0,255,0),2)


cv2.imshow("Find Faces!", image)

cv2.waitKey(0)