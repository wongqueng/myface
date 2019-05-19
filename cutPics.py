import cv2
import numpy as np
import os.path

input_data_path = "D:\PycharmProjects\myface\AndyLou_IMAGES/"
save_path = "D:/PycharmProjects/myface/train/AndyLou/"
cascade_path = 'D:/PycharmProjects/myface\haarcascades/haarcascade_frontalface_alt.xml'
faceCascade = cv2.CascadeClassifier(cascade_path)
# faceCascade.load('D:/PycharmProjects/myface/haarcascades/haarcascade_frontalface_alt.xml')
image_count = 0

face_detect_count = 0

for i in range(image_count):
    if os.path.isfile(input_data_path  + str(i) + '.jpg'):
        try:
            img = cv2.imread(input_data_path + str(i) + '.jpg', cv2.IMREAD_COLOR)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            face = faceCascade.detectMultiScale(gray, 1.1, 3)

            if len(face) > 0:
                for rect in face:
                    x = rect[0]
                    y = rect[1]
                    w = rect[2]
                    h = rect[3]

                    cv2.imwrite(save_path + 'face-' + str(face_detect_count) + '.jpg', img[y:y + h, x:x + w])
                    face_detect_count = face_detect_count + 1
            else:
                print('image' + str(i) + ': No Face')
        except Exception as e:
            print('image' + str(i) + ': Exception - ' + str(e))
    else:
        print('image' + str(i) + ': No File')