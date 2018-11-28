# -*- coding: utf-8 -*-
"""
Created on Tue Nov 27 21:01:45 2018

@author: MR
"""

import tensorflow as tf
import cifar10_input
import numpy as np
#from tensorflow.contrib.layers.python.layers import batch_norm
#from matplotlib import pylab

batch_size=128
data_dir='.\cifar-10-python\cifar-10-batches-py'
image_train,labels_train=cifar10_input.inputs(eval_data=False,data_dir=data_dir,batch_size=batch_size)
image_test,labels_test=cifar10_input.inputs(eval_data=True,data_dir=data_dir,batch_size=batch_size)

#定义batch_norm层，传参里面还包含一个是否训练
#def batch_norm_layer(value,train=None,name='batch_norm'):
#    if train is not None:
#        return batch_norm(value,decay=0.9,updates_collections=None,is_training=True)
#    else:
#        return batch_norm(value,decay=0.9,updates_collections=None,is_training=False)

#输入输出
x=tf.placeholder(tf.float32,[None,24,24,3])
y=tf.placeholder(tf.float32,[None,10])
train=tf.placeholder(tf.float32)

x_image=tf.reshape(x,[-1,24,24,3])


h_conv1=tf.layers.conv2d(x_image,32,[3,3],1,padding='SAME',activation=tf.nn.relu)
bn_1=tf.layers.batch_normalization(h_conv1,training=True)
h_pool1=tf.contrib.layers.max_pool2d(bn_1,[2,2],stride=2,padding='SAME')

h_conv2=tf.layers.conv2d(h_pool1,32,[3,3],1,padding='SAME',activation=tf.nn.relu)
bn2=tf.layers.batch_normalization(h_conv2,training=True)
h_pool2=tf.contrib.layers.max_pool2d(bn2,[2,2],stride=2,padding='SAME')

nt_hpool2=tf.contrib.layers.avg_pool2d(h_pool2,[6,6],stride=6,padding='SAME')

flatten_layer=tf.contrib.layers.flatten(nt_hpool2,'flatten_layer')

#nt_hpool2_flat=tf.reshape(nt_hpool2,[-1,32])

logits_out=tf.contrib.layers.fully_connected(flatten_layer,10,activation_fn=tf.nn.softmax)

#交叉熵
cross_entropy=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits_out,labels=y))

global_step=tf.Variable(0)
decaylearning_rate=tf.train.exponential_decay(learning_rate=0.1,global_step=global_step,decay_steps=5000,decay_rate=0.1,staircase=True)

#opt=tf.train.AdamOptimizer(learning_rate=decaylearning_rate)
opt = tf.train.AdadeltaOptimizer(learning_rate=decaylearning_rate, name='optimizer')

correct_prediction=tf.equal(tf.argmax(logits_out,1),tf.argmax(y,1))
accuracy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
#使用了bath_norm还要对其中的ops进行更新
update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
with tf.control_dependencies(update_ops):
    grads=opt.compute_gradients(cross_entropy)
    train_op = opt.apply_gradients(grads,global_step=global_step)
    
#一些与设置有关的参数
tf_config = tf.ConfigProto()
tf_config.gpu_options.allow_growth = True
tf_config.allow_soft_placement = True
print('启动session')
#启动session
sess = tf.InteractiveSession(config=tf_config)
sess.run(tf.global_variables_initializer())

for i in range(10000):
    print('第%d次'%i)
    image_batch_xs,labels_batch_ys=sess.run([image_train,labels_train])
    print(image_batch_xs.shape,labels_batch_ys.shape)
    label_onehot=tf.one_hot(labels_batch_ys,10) 
    label_onehot=label_onehot.eval(session=sess)
    #print(label_onehot)
    #label_onehot=np.eye(10,dtype=float)[labels_batch] #将标签转化为onehot编码
    _,c,step,loss,acc=sess.run([opt,global_step,cross_entropy,accuracy],feed_dict={x:image_batch_xs,y:label_onehot})
    #train_step.run(feed_dict={x:image_batch,y:label_onehot},session=sess)
    if i%200==0:
        log_str = 'step:%d \t loss:%.6f \t acc:%.6f' % (step, loss, acc)
        tf.logging.info(log_str)
        
#image_batch,label_batch=sess.run([image_test,labels_test])
#label_onehot=np.eye(10,dtype=float)[labels_test]
#accuracy_test=accuracy.eval(feed_dict={x:image_batch,y:label_onehot},session=sess)
#print('test accuracy %g'%train_accuarcy)