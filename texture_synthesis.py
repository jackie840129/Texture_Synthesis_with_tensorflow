import numpy as np
import match as m
import scipy
import tensorflow as tf
import skimage 
import vgg19
import utils 
import matplotlib.pyplot as plt
import time
import histogram as h
im = skimage.io.imread("./test_data/2.jpg")
img1 = utils.load_image("./test_data/2.jpg")
skimage.io.imsave("origin.jpg",img1)
print(im.shape)
#with tf.device('/gpu:1'):
batch1 = img1.reshape((1, 256, 256, 3))
images = tf.Variable(batch1,trainable=False,dtype='float64')
images = tf.cast(images,tf.float32)

change = h.origin_his(im)
skimage.io.imsave("noise.jpg",change)
init = utils.load_image("noise.jpg")
batch2 = init.reshape((1,256,256,3))
batch2 = batch2.astype("float32")
tex = tf.Variable(batch2)

#init = np.random.uniform(0.,1.,(256,256,3))
#init = m.histogram_matching(init,img1)

#tex = tf.Variable(init.reshape(1,256,256,3),dtype=tf.float32)
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.4   
with tf.Session(config=config) as sess:
    vgg = vgg19.Vgg19()
    vgg2 = vgg19.Vgg19()
    with tf.name_scope("o"):
        vgg.build(images)
    with tf.name_scope("n"):
        vgg2.build(tex)
    loss_sum = 0.
    for i in range(0,5):     
        origin = vgg.conv_list[i]
        new = vgg2.conv_list[i]
        shape = origin.get_shape().as_list()
        N = shape[3]
        M = shape[1]*shape[2]
        F = tf.reshape(origin,(-1,N))
        Gram_o = (tf.matmul(tf.transpose(F),F)/(N*M))
        F_t = tf.reshape(new,(-1,N))
        Gram_n = tf.matmul(tf.transpose(F_t),F_t)/(N*M)
        loss = tf.nn.l2_loss((Gram_o-Gram_n))/2
        loss_sum = tf.add(loss_sum,loss)
    train_step=tf.train.AdamOptimizer(3e-3).minimize(loss_sum, var_list=[tex])
    train_step2=tf.train.AdamOptimizer(1e-3).minimize(loss_sum, var_list=[tex])
    train_step3=tf.train.AdamOptimizer(0.8e-3).minimize(loss_sum, var_list=[tex])
    train_step4=tf.train.AdamOptimizer(0.08e-3).minimize(loss_sum, var_list=[tex])
    
    c = tf.maximum(0., tf.minimum(1., tex))
    ass = tex.assign(c)

    sess.run(tf.initialize_all_variables())
    
    for i in range(0,5000):
        if i<500:
            sess.run(train_step)
        elif i>=500 and i<1000:
            sess.run(train_step2)
        elif i>=1000 and i<2000:
            sess.run(train_step3)
        elif i>=2000:
            sess.run(train_step4)
        sess.run(ass)
        if i % 10 == 0:
            print("loss of iteration ",i,loss_sum.eval(),sep=" ") 
        if i%500 == 0:
            answer = tex.eval()
            answer = answer.reshape(256,256,3)
            print('Mean = ', np.mean(answer))
            d = "./answer/"
            a = str(i)
            b = "after.jpg"
            c=d+a+b
            skimage.io.imsave(c,answer)
        #answer = answer.reshape(1,256,256,3)
        #tex.assign(answer)
        
    answer=tex.eval()
    answer=answer.reshape(256,256,3)
    answer = m.histogram_matching(answer,img1)
    answer.astype(np.float64)
    print(answer)
    skimage.io.imsave("./answer/after.jpg",answer)
