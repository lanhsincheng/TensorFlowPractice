''''''''''

add_layer function use

'''''''''''
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt
#one_hot means one is hot and rest are off
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)
''''''''''
0 = [1,0,0,0,0,0,0,0,0]
1 = [0,1,0,0,0,0,0,0,0]
2 = [0,0,1,0,0,0,0,0,0]
3 = [0,0,0,1,0,0,0,0,0]
... 
'''''''''''

#define hm nodes per layer/ hm class/ batch size
n_nodes_hl1 = 784
n_nodes_hl2 = 500
n_nodes_hl3 = 500
n_classes = 10
batch_size = 100

#define placeholder of computational graph
with tf.name_scope('inputs'):
    x = tf.placeholder(tf.float32, shape=(None,784), name='x_input') # None means we dont care how many samples 28*28=784
    y = tf.placeholder(tf.float32, shape=(None,10), name='y_input') # 10 outpot(10 classes)

#define computational graph

# method1: by using add_layer
def add_layer(input, in_size, out_size,n_layer, activation_funct=None): #activation function is none means is linear function
    layer_name = "layer%s" % n_layer # % can do some formating operation
    with tf.name_scope('layer'):

        with tf.name_scope('weight'):
            weight_array = tf.Variable(tf.random_normal([in_size,out_size])) # [i, o] i col and o row
            tf.summary.histogram(layer_name + '/weights',weight_array)

        with tf.name_scope('biase'):
            #biase_list = tf.Variable(tf.zeros[1,out_size]+0.1)
            biase_list = tf.Variable(tf.random_normal([out_size]))
            tf.summary.histogram(layer_name + '/biase', biase_list)

        with tf.name_scope('wx_plus_b'):
            wx_plus_b = tf.matmul(input,weight_array) + biase_list

        if activation_funct is None:
            output = wx_plus_b
        else:
            output = activation_funct(wx_plus_b)
        tf.summary.histogram(layer_name + '/output', output)

    return output

# start to construct layer
def built(x):
    l1 = add_layer(x, 784, n_nodes_hl1, 0, tf.nn.relu )
    l2 = add_layer(l1, n_nodes_hl1, n_nodes_hl2, 1, tf.nn.relu)
    l3 = add_layer(l2, n_nodes_hl2, n_nodes_hl3, 2,  tf.nn.relu)
    output = add_layer(l3, n_nodes_hl3, n_classes, 3, None)

    return output

output = built(x)

with tf.name_scope('loss'):
    cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits_v2(logits=output, labels= y) )
    tf.summary.scalar('loss', cost)

with tf.name_scope('optimizer'):
    optimizer = tf.train.AdamOptimizer().minimize(cost) #this will adjust cost variable

#start to evaluate our prediction
with tf.name_scope('accuracy'):
    correct = tf.equal(tf.argmax(output, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct,'float')) #conver [T T T F F ....] into [1 1 1 0 0 ....]
    tf.summary.scalar('accuracy', accuracy)

hm_epoch = 10
init = tf.global_variables_initializer()
with tf.Session() as sess:
    merged = tf.summary.merge_all()
    writer = tf.summary.FileWriter("logs/",graph=sess.graph)
    sess.run(init)
    for epoch in range(hm_epoch):
        #training
        epoch_loss = 0
        for _ in range(int(mnist.train.num_examples / batch_size)):
            epoch_x, epoch_y = mnist.train.next_batch(batch_size)
            _, c = sess.run( [optimizer, cost], feed_dict={x:epoch_x, y:epoch_y}) #we have to run optimuzer and cost
            epoch_loss+=c
        print('epoch: ',epoch,'total epoch: ',hm_epoch,'loss:', epoch_loss )

        if epoch % 2 == 0:
            res = sess.run(merged,feed_dict={x:mnist.train.images, y:mnist.train.labels})
            writer.add_summary(res,epoch)

    print('method 1 accuracy:', accuracy.eval( {x: mnist.test.images, y: mnist.test.labels})) # accuracy on test set

