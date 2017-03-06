import numpy as np
import scipy
import tensorflow as tf
import skimage 
import vgg19
import utils 
import time
import os
import sys
import shutil

img_name = str(sys.argv[1])
if os.path.exists('./result'):
    shutil.rmtree('./result')
else :
    os.makedirs('./result')

## Process the origin image to be an tensorflow tensor
origin_img  = skimage.io.imread("./test_data/"+img_name) # [0,255]
proc_img = utils.load_image("./test_data/"+img_name)   # [0,1)
skimage.io.imsave("result/origin.jpg",proc_img)
batch1 = proc_img.reshape((1, 256, 256, 3))
images = tf.Variable(batch1,trainable=False,dtype='float64')
images = tf.cast(images,tf.float32)
#####################################################
## Generate a noise img depends on the histogram of origin img 
img  = skimage.io.imread("./result/origin.jpg") # [0,255]
noise = utils.his_noise(img)
skimage.io.imsave("result/his_noise.jpg",noise)
init = utils.load_image("result/his_noise.jpg") #[0,1)
batch2 = init.reshape((1,256,256,3)).astype("float32")
tex = tf.Variable(batch2)

#####################################################
#1.
config = tf.ConfigProto()
# config.gpu_options.per_process_gpu_memory_fraction = 0.4   
#2.
os.environ["CUDA_VISIBLE_DEVICEs"]="0"

with tf.Session(config=config) as sess:
    vgg = vgg19.Vgg19()
    vgg2 = vgg19.Vgg19()
    with tf.name_scope("origin"):
        vgg.build(images)
    with tf.name_scope("new"):
        vgg2.build(tex)
    ## Caculate the Loss according to the paper
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
    train_step=tf.train.AdamOptimizer(0.01).minimize(loss_sum, var_list=[tex])
    # train_step2=tf.train.AdamOptimizer(1e-3).minimize(loss_sum, var_list=[tex])
    # train_step3=tf.train.AdamOptimizer(0.8e-3).minimize(loss_sum, var_list=[tex])
    # train_step4=tf.train.AdamOptimizer(0.08e-3).minimize(loss_sum, var_list=[tex])
    
    restrict = tf.maximum(0., tf.minimum(1., tex))
    r_tex = tex.assign(restrict)

    sess.run(tf.initialize_all_variables())
    
    Iteration = 5000
    for i in range(0,Iteration):
        # if i<500:
            # sess.run(train_step)
        # elif i>=500 and i<1000:
            # sess.run(train_step2)
        # elif i>=1000 and i<2000:
            # sess.run(train_step3)
        # elif i>=2000:
            # sess.run(train_step4)
        sess.run(train_step)
        sess.run(r_tex)
        if i%10==0:
            loss = loss_sum.eval()
            sys.stdout.write('\r')
            sys.stdout.write("[%-50s] %d/%d ,loss=%.4f" % ('='*int(i*50/Iteration),i,Iteration,loss))
            sys.stdout.flush()
            # sleep(0.25)
        # if i % 10 == 0:
            # print("loss of iteration ",i,loss_sum.eval(),sep=" ") 
        if i%500 == 0 and i!=0:
            answer = tex.eval()
            answer = answer.reshape(256,256,3)
            answer = (answer*255).astype('uint8')
            # print('Mean = ', np.mean(answer))
            filename = "./result/%safter.jpg"%(str(i))
            skimage.io.imsave(filename,answer)
        
    answer=tex.eval()
    answer=answer.reshape(256,256,3)
    answer = (utils.histogram_matching(answer,proc_img)*255.).astype('uint8')
    skimage.io.imsave("./result/final.jpg",answer)
