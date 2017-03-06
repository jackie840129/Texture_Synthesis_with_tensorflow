# Texture_Synthesis_with_tensorflow
This is an undergraduate research project that try to synthesis textures by using convolutional nerual networks with toolkit Tensorflow.

The algorithm and the origin paper is from "Texture Synthesis Using Convolutional Neural Networks" (Gatys et al., NIPS 2015) (http://arxiv.org/abs/1505.07376).

## Algorithm in the paper
![alt tag](http://i.imgur.com/r0sHxNs.png)

**Briefly explain**: Let a target image and a noise image go through two same pretrained VGG-19 network, then use gradient descent to minimize the difference between the features in the CNN network.

### Changes
I use histogram matching first to create a noise picture which has the same distribution as the origin image.

## Results

origin ![image](http://i.imgur.com/hlfckUu.png = 50x50) 

synthesis ![alt tag](http://i.imgur.com/s3zyPM7.png)
