import sys
import numpy as np
import cv2
import tensorflow as tf
import os
import random
import train

cascade_path = 'D:/PycharmProjects/myface\haarcascades/haarcascade_frontalface_alt.xml'
faceCascade = cv2.CascadeClassifier(cascade_path)

HUMAN_NAMES = {
  0: u"刘德华",
  1: u"吴彦祖"
}

def evaluation(img_path, ckpt_path):
  tf.reset_default_graph()

  f = open(img_path, 'r')
  img = cv2.imread(img_path, cv2.IMREAD_COLOR)

  gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  face = faceCascade.detectMultiScale(gray, 1.1, 3)

  if len(face) > 0:
    for rect in face:
      random_str = str(random.random())

      cv2.rectangle(img, tuple(rect[0:2]), tuple(rect[0:2]+rect[2:4]), (0, 0, 255), thickness=2)

      face_detect_img_path = 'D:/PycharmProjects/myface/eval_images/' + random_str + '.jpg'

      cv2.imwrite(face_detect_img_path, img)
      x = rect[0]
      y = rect[1]
      w = rect[2]
      h = rect[3]

      cv2.imwrite('D:/PycharmProjects/myface/eval_images/' + random_str + '.jpg', img[y:y+h, x:x+w])

      target_image_path = 'D:/PycharmProjects/myface/eval_images/' + random_str + '.jpg'
  else:
    print('image:No Face')
    return
  f.close()
  f = open(target_image_path, 'r')

  image = []
  img = cv2.imread(target_image_path)
  img = cv2.resize(img, (28, 28))

  image.append(img.flatten().astype(np.float32)/255.0)
  image = np.asarray(image)

  logits = train.inference(image, 1.0)

  sess = tf.InteractiveSession()

  saver = tf.train.Saver()

  sess.run(tf.global_variables_initializer())

  if ckpt_path:
    saver.restore(sess, ckpt_path)

  softmax = logits.eval()

  result = softmax[0]

  rates = [round(n * 100.0, 1) for n in result]
  humans = []

  for index, rate in enumerate(rates):
    name = HUMAN_NAMES[index]
    humans.append({
      'label': index,
      'name': name,
      'rate': rate
    })

  rank = sorted(humans, key=lambda x: x['rate'], reverse=True)

  print(img_path)
  print(rank)

  return [rank, os.path.basename(img_path), random_str + '.jpg']

if __name__ == '__main__':
  TEST_IMAGE_PATHS = [ 'face-0.jpg', 'face-4.jpg','face-6.jpg','face-16.jpg','face-41.jpg', ]
  for image_path in TEST_IMAGE_PATHS:
    evaluation('D:/PycharmProjects/myface/eval_images/'+image_path, 'D:/PycharmProjects/myface/model/model.ckpt')