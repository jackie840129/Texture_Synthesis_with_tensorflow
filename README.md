# Texture_Synthesis_with_tensorflow    make:Jan.2016
This is an undergraduate research project that try to synthesis textures by using convolutional nerual networks with toolkit Tensorflow.

The algorithm and the origin paper is from "Texture Synthesis Using Convolutional Neural Networks" (Gatys et al., NIPS 2015) (http://arxiv.org/abs/1505.07376).

## Algorithm in the paper
![alt tag](http://i.imgur.com/r0sHxNs.png)

**Briefly explain**: Let a target image and a noise image go through two same pretrained VGG-19 network, then use gradient descent to minimize the difference between the features in the CNN network.

### Changes
I use histogram matching first to create a noise picture which has the same distribution as the origin image.

## Usage
### Require
**Tensorflow**: version : **0.8.0** !!

**Python package:** Numpy, Scipy, skimage 

### Data
The VGG-19 NPY file can be downloaded from [here](https://mega.nz/#!xZ8glS6J!MAnE91ND_WyfZ_8mvkuSa2YcA7q-1ehfSm-Q1fxOvvs)

### Instruction
`$ python texture_synthesis.py pebbles.jpg`

(Put your images in **test_data/** folder and directly type the image name)

**The output images will be generated in result/ directory,which contains many images with different phase**

## Results
### Pebbles
**origin**
<img src="http://i.imgur.com/hlfckUu.png" width="256" height="256">
**synthetis**
<img src="http://i.imgur.com/s3zyPM7.png" width="256" height="256">

### Bricks
**origin**
<img src="http://i.imgur.com/bxwaRFA.png" width="256" height="256">
**synthetis**
<img src="http://i.imgur.com/QpiRrPo.png" width="256" height="256">
