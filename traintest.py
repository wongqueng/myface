import tensorflow as tf

#定义隐藏层参数，每个w变量是一个tensor(可以当成是n*m的数组，n表示上一层结点个数，m表示本层结点个数)表示上一层与本层的连接权重,这里先随机定义权重
w1=tf.Variable(tf.random_normal([2,3],stddev=1))
w2=tf.Variable(tf.random_normal([3,1],stddev=1))

#定义存放输入数据的地方，也就是x向量,这里shape为前一个传入训练的样本个数，后面出入每个样本的维度大小
x=tf.placeholder(tf.float32,shape=(None,2),name="input")
#矩阵乘法
a=tf.matmul(x,w1)
y=tf.matmul(a,w2,name="output")

with tf.Session() as sess:
    #新版本好像不能用这个函数初始化所有变量了
    init_op=tf.global_variables_initializer()
    sess.run(init_op)
    #feed_dict用于向y中的x传入参数，这里传入3个，则y输出为一个3*1的tensor
    print(sess.run(y,feed_dict={x:[[0.7,0.9],[1.0,1.5],[2.1,2.3]]}))
    output_graph_def = tf.graph_util.convert_variables_to_constants(sess,
                                                                    sess.graph_def, output_node_names=["output"])
    with tf.gfile.FastGFile('D:/PycharmProjects/myface/model/test.pb', mode="wb") as f:
        f.write(output_graph_def.SerializeToString());