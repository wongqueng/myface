import sys
import cv2
import random
import numpy as np
import tensorflow as tf

NUM_CLASSES = 2

IMAGE_SIZE = 28

IMAGE_PIXELS = IMAGE_SIZE * IMAGE_SIZE * 3
MAX_STEPS=6
BATCH_SIZE=40
LEARNING_RATE=1e-4


flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string('train', 'D:/PycharmProjects/myface/train/data.txt', 'File name of train data')

flags.DEFINE_string('test', 'D:/PycharmProjects/myface/train/test_data.txt', 'File name of test data')

flags.DEFINE_string('train_dir', 'D:/PycharmProjects/myface/train/data/', 'Directory to put the training data')

flags.DEFINE_integer('max_steps', 6, 'Number of steps to run trainer.')

flags.DEFINE_integer('batch_size', 30, 'Batch size Must divide evenly into the dataset sizes.')

flags.DEFINE_float('learning_rate', 1e-4, 'Initial learning rate.')


def weight_variable(shape):
		# 正态分布，标准差为0.1，默认最大为1，最小为-1，均值为0
    	initial = tf.truncated_normal(shape, stddev=0.1)
    	return tf.Variable(initial)
def bias_variable(shape):
		# 创建一个结构为shape矩阵也可以说是数组shape声明其行列，初始化所有值为0.1
    	initial = tf.constant(0.1, shape=shape)
    	return tf.Variable(initial)
def conv2d(x, W):
		# 卷积遍历各方向步数为1，SAME：边缘外自动补0，遍历相乘
  	return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
def max_pool_2x2(x):
		# 池化卷积结果（conv2d）池化层采用kernel大小为2*2，步数也为2，周围补0，取最大值。数据量缩小了4倍
  	return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1], padding='SAME')
xs = tf.placeholder(tf.float32, [None, 28 * 28*3],name="input")
        # 类别是0-9总共10个类别，对应输出分类结果
ys = tf.placeholder(tf.float32, [None, 2])
keep_prob = tf.placeholder(tf.float32)
# x_image又把xs reshape成了28*28*1的形状，因为是灰色图片，所以通道是1.作为训练时的input，-1代表图片数量不定
x_image = tf.reshape(xs, [-1, 28, 28, 3])

# 三，搭建网络,定义算法公式，也就是forward时的计算

## 第一层卷积操作 ##
# 第一二参数值得卷积核尺寸大小，即patch，第三个参数是图像通道数，第四个参数是卷积核的数目，代表会出现多少个卷积特征图像;
W_conv1 = weight_variable([5, 5,3, 32])
# 对于每一个卷积核都有一个对应的偏置量。
b_conv1 = bias_variable([32])
# 图片乘以卷积核，并加上偏执量，卷积结果28x28x32
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
# 池化结果14x14x32 卷积结果乘以池化卷积核
h_pool1 = max_pool_2x2(h_conv1)

## 第二层卷积操作 ##
# 32通道卷积，卷积出64个特征
w_conv2 = weight_variable([5, 5, 32, 64])
# 64个偏执数据
b_conv2 = bias_variable([64])
# 注意h_pool1是上一层的池化结果，#卷积结果14x14x64
h_conv2 = tf.nn.relu(conv2d(h_pool1, w_conv2) + b_conv2)
# 池化结果7x7x64
h_pool2 = max_pool_2x2(h_conv2)
# 原图像尺寸28*28，第一轮图像缩小为14*14，共有32张，第二轮后图像缩小为7*7，共有64张

## 第三层全连接操作 ##
# 二维张量，第一个参数7*7*64的patch，也可以认为是只有一行7*7*64个数据的卷积，第二个参数代表卷积个数共1024个
W_fc1 = weight_variable([7 * 7 * 64, 1024])
# 1024个偏执数据
b_fc1 = bias_variable([1024])
# 将第二层卷积池化结果reshape成只有一行7*7*64个数据# [n_samples, 7, 7, 64] ->> [n_samples, 7*7*64]
h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
# 卷积操作，结果是1*1*1024，单行乘以单列等于1*1矩阵，matmul实现最基本的矩阵相乘，不同于tf.nn.conv2d的遍历相乘，自动认为是前行向量后列向量
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

# dropout操作，减少过拟合，其实就是降低上一层某些输入的权重scale，甚至置为0，升高某些输入的权值，甚至置为2，防止评测曲线出现震荡，个人觉得样本较少时很必要
# 使用占位符，由dropout自动确定scale，也可以自定义，比如0.5，根据tensorflow文档可知，程序中真实使用的值为1/0.5=2，也就是某些输入乘以2，同时某些输入乘以0
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)  # 对卷积结果执行dropout操作

## 第四层输出操作 ##
# 二维张量，1*1024矩阵卷积，共10个卷积，对应我们开始的ys长度为10
W_fc2 = weight_variable([1024, 2])
b_fc2 = bias_variable([2])
# 最后的分类，结果为1*1*10 softmax和sigmoid都是基于logistic分类算法，一个是多分类一个是二分类
y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2,name="outputt")

# 四，定义loss(最小误差概率)，选定优化优化loss，
cross_entropy = -tf.reduce_sum(ys * tf.log(y_conv))  # 定义交叉熵为loss函数
train_step = tf.train.AdamOptimizer(0.5).minimize(
    cross_entropy)  # 调用优化器优化，其实就是通过喂数据争取cross_entropy最小化

# 五，开始数据训练以及评测
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(ys, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

f = open(FLAGS.train, 'r')
train_image = []
train_label = []

for line in f:
    line = line.rstrip()
    l = line.split()

    img = cv2.imread(l[0])
    img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))

    train_image.append(img.flatten().astype(np.float32) / 255.0)

    tmp = np.zeros(NUM_CLASSES)
    tmp[int(l[1])] = 1
    train_label.append(tmp)

train_image = np.asarray(train_image)
train_label = np.asarray(train_label)
f.close()

f = open(FLAGS.test, 'r')
test_image = []
test_label = []
for line in f:
    line = line.rstrip()
    l = line.split()
    img = cv2.imread(l[0])
    img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
    test_image.append(img.flatten().astype(np.float32) / 255.0)
    tmp = np.zeros(NUM_CLASSES)
    tmp[int(l[1])] = 1
    test_label.append(tmp)
test_image = np.asarray(test_image)
test_label = np.asarray(test_label)
f.close()
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for step in range(FLAGS.max_steps):
        for i in range(int(len(train_image) / FLAGS.batch_size)):
            batch = FLAGS.batch_size * i

            sess.run(train_step, feed_dict={
                xs: train_image[batch:batch + FLAGS.batch_size],
                ys: train_label[batch:batch + FLAGS.batch_size],
                keep_prob: 0.5})

        train_accuracy = sess.run(accuracy, feed_dict={
            xs: train_image,
            ys: train_label,
            keep_prob: 1.0})
        print("step %d, training accuracy %g" % (step, train_accuracy))
    output_graph_def = tf.graph_util.convert_variables_to_constants(sess,
                                                                    sess.graph_def, output_node_names=["outputt"])
    with tf.gfile.FastGFile('D:/PycharmProjects/myface/model/nypb.pb', mode="wb") as f:
        f.write(output_graph_def.SerializeToString());

















