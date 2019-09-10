''''''''''

add layer by hand without function use

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
#method2: add by hand
def neural_network_model(data):

    hidden_l_layer = {
        'weight': tf.Variable(tf.random_normal([784,n_nodes_hl1])),
        'bias': tf.Variable(tf.random_normal([n_nodes_hl1]))
    }


    hidden_2_layer = {
        'weight': tf.Variable(tf.random_normal([n_nodes_hl1, n_nodes_hl2])),
        'bias': tf.Variable(tf.random_normal([n_nodes_hl2]))
    }


    hidden_3_layer = {
        'weight': tf.Variable(tf.random_normal([n_nodes_hl2, n_nodes_hl3])),
        'bias': tf.Variable(tf.random_normal([n_nodes_hl3]))
    }


    output_layer = {
        'weight': tf.Variable(tf.random_normal([n_nodes_hl3, n_classes])),
        'bias': tf.Variable(tf.random_normal([n_classes]))
    }

    # go through activation func
    l1 = tf.add( tf.matmul(data, hidden_l_layer['weight']) , hidden_l_layer['bias'])
    l1 = tf.nn.relu(l1)

    l2 = tf.add(tf.matmul(l1, hidden_2_layer['weight']) , hidden_2_layer['bias'])
    l2 = tf.nn.relu(l2)

    l3 = tf.add(tf.matmul(l2, hidden_3_layer['weight']) , hidden_3_layer['bias'])
    l3 = tf.nn.relu(l3)

    output = tf.matmul(l3, output_layer['weight']) + output_layer['bias']
    return  output #output is an one-hot array [00010000....  ]


def train_neural_network(x):
    prediction = neural_network_model(x) # x is placeholder and will feed data later

    cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits_v2(logits=prediction, labels= y) )
    optimizer = tf.train.AdamOptimizer().minimize(cost) #this will adjust cost variable

    # training process
    hm_epoch = 10
    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)
        for epoch in range(hm_epoch):
            epoch_loss = 0
            for _ in range(int(mnist.train.num_examples/batch_size)):
                epoch_x, epoch_y = mnist.train.next_batch(batch_size)
                _, c = sess.run( [optimizer, cost], feed_dict={x:epoch_x, y:epoch_y}) #we have to run optimuzer and cost
                epoch_loss+=c
            print('epoch: ',epoch,'total epoch: ',hm_epoch,'loss:', epoch_loss )

        #start to evaluate our prediction
        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct,'float')) #conver [T T T F F ....] into [1 1 1 0 0 ....]
        print('method 2 accuracy:', accuracy.eval( {x: mnist.test.images, y: mnist.test.labels})) # accuracy on test set

train_neural_network(x)