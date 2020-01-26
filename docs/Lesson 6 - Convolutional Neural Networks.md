# 6 - Convolutional Neural Networks
### Lectures
#### Applications of CNNs
* [WaveNet](https://deepmind.com/blog/wavenet-generative-model-raw-audio/)
* [Text Classification](http://www.wildml.com/2015/12/implementing-a-cnn-for-text-classification-in-tensorflow/)
* [Language Translation](https://code.facebook.com/posts/1978007565818999/a-novel-approach-to-neural-machine-translation/)
* [Play Atari games](https://deepmind.com/research/dqn/)
* [Play Pictionary](https://quickdraw.withgoogle.com/#)
* [Play Go](https://deepmind.com/research/alphago/)
* [CNNs powered Drone](https://www.youtube.com/watch?v=wSFYOw4VIYY)
* Self-Driving Car
* [Predict depth from a single image](https://www.cs.nyu.edu/~deigen/depth/)
* [Localize breast cancer](https://research.googleblog.com/2017/03/assisting-pathologists-in-detecting.html)
* [Save endangered species](https://blogs.nvidia.com/blog/2016/11/04/saving-endangered-species/?adbsc=social_20170303_70517416)
* [Face App](http://www.digitaltrends.com/photography/faceapp-neural-net-image-editing/)

#### Lesson Outline
* About CNN (Convolutional Neural Network) and how they improve our ability to classify images
* How CNN identify features and how CNN can be used for image classification
* Various layer that make up a complete CNN
* __A feature__ is to think about what we are visually drawn to when we first see an object and when we identify different objects. For example what do we look at to distinguish a cat and a dog? The shape of the eyes, the size, and how they move

#### MNIST Dataset
<p align="center">
  <img src="./images/lesson-5/mnist-database.PNG" width="50%">
</p>

* Most famous database

<p align="center">
  <img src="./images/lesson-5/mnist.png" width="50%">
</p>

#### How Computers Interpret Images
<p align="center">
  <img src="./images/lesson-5/normalization.PNG" width="50%">
</p>

* __Data normalization__ is an important pre-processing step. It ensures that each input (each pixel value, in this case) comes from a standard distribution. 

* [Normalize transformation in PyTorch](https://pytorch.org/docs/stable/torchvision/transforms.html#transforms-on-torch-tensor)

<p align="center">
  <img src="./images/lesson-5/flattening.PNG" width="50%">
</p>

#### MLP (Multi Layer Perceptron) Structure & Class Scores
<p align="center">
  <img src="./images/lesson-5/mlp.PNG" width="50%">
</p>
* layer

A set of neurons in a neural network that process a set of input features, or the output of those neurons.

Layers are Python functions that take Tensors and configuration options as input and produce other tensors as output. Once the necessary Tensors have been composed, the user can convert the result into an Estimator via a model function.

<p align="center">
  <img src="./images/lesson-5/class-scores.PNG" width="50%">
</p>

* class

One of a set of enumerated target values for a label. For example, in a binary classification model that detects spam, the two classes are spam and not spam. In a multi-class classification model that identifies dog breeds, the classes would be poodle, beagle, pug, and so on.

* scoring

The part of a recommendation system that provides a value or ranking for each item produced by the candidate generation phase.

#### Do Your Research
* More hidden layers generally means more ability to recognize complex pattern
* One or two hidden layers should work fine for small images
* Keep looking for a resource or two that appeals to you
* Try out the models in code

<p align="center">
  <img src="./images/lesson-5/do-your-research.PNG" width="50%">
</p>

#### Loss & Optimization

<p align="center">
  <img src="./images/lesson-5/learn-from-mistakes.PNG" width="50%">
</p>

<p align="center">
  <img src="./images/lesson-5/cross-entropy-loss.PNG" width="50%">
</p>

<p align="center">
  <img src="./images/lesson-5/gradient-descent.PNG" width="50%">
</p>

#### Defining a Network in PyTorch
* Rectified Linear Unit (ReLU)

An activation function with the following rules:
  * If input is negative or zero, output is 0.
  * If input is positive, output is equal to input.

<p align="center">
  <img src="./images/lesson-5/relu-ex.png" width="50%">
</p>

#### Training the Network
The steps for training/learning from a batch of data are described in the comments below:

1. Clear the gradients of all optimized variables
2. Forward pass: compute predicted outputs by passing inputs to the model
3. Calculate the loss
4. Backward pass: compute gradient of the loss with respect to model parameters
5. Perform a single optimization step (parameter update)
6. Update average training loss

#### One Solution
* `model.eval()` will set all the layers in your model to evaluation mode. 
* This affects layers like dropout layers that turn "off" nodes during training with some probability, but should allow every node to be "on" for evaluation. 
* So, you should set your model to evaluation mode before testing or validating your model and set it to `model.train()` (training mode) only during the training loop.

#### Model Validation
<p align="center">
  <img src="./images/lesson-5/model-validation.PNG" width="50%">
</p>

<p align="center">
  <img src="./images/lesson-5/early-stopping.PNG" width="50%">
</p>

#### Validation Loss
* We create a validation set to:
  1. Measure how well a model generalizes, during training
  2. Tell us when to stop training a model; when the validation loss stops decreasing (and especially when the validation loss starts increasing and the training loss is still decreasing)

#### Image Classification Steps
<p align="center">
  <img src="./images/lesson-5/image-classification-steps.PNG" width="50%">
</p>

#### MLPs vs CNNs
* MNIST already centered, real image can be any position
<p align="center">
  <img src="./images/lesson-5/mnist-vs-real.PNG" width="50%">
</p>

#### Local Connectivity
* Difference between MLP vs CNN
<p align="center">
  <img src="./images/lesson-5/mlp-vs-cnn.PNG" width="50%">
</p>

* Sparsely connected layer
<p align="center">
  <img src="./images/lesson-5/local-conn.PNG" width="50%">
</p>

#### Filters and the Convolutional Layer
* CNN is special kind of NN that can remember spatial information
* The key to remember spatial information is convolutional layer, which apply series of different image filters (convolutional kernels) to input image

<p align="center">
  <img src="./images/lesson-5/filtered-images.PNG" width="50%">
</p>

* CNN should learn to identify spatial patterns like curves and lines that make up number six

<p align="center">
  <img src="./images/lesson-5/conv-layer.PNG" width="50%">
</p>

#### Filters & Edges
* Intensity is a measure of light and dark, similiar to brightness
* To identify the edges of an object, look at abrupt changes in intensity
* Filters

  To detect changes in intensity in an image, look at groups of pixels and react to alternating patterns of dark/light pixels. Producing an output that shows edges of objects and differing textures.

* Edges

  Area in images where the intensity changes very quickly

#### Frequency in Images

<p align="center">
  <img src="./images/lesson-5/hf-image.png" width="50%">
</p>

* Frequency in images is a __rate of change__.
  * on the scarf and striped shirt, we have a high-frequency image pattern
  * parts of the sky and background that change very gradually, which is considered a smooth, low-frequency pattern
* __High-frequency components__ also correspond to __the edges__ of objects in images, which can help us classify those objects.

#### High-pass Filters
<p align="center">
  <img src="./images/lesson-5/filters.PNG" width="50%">
</p>
<p align="center">
  <img src="./images/lesson-5/high-pass filters.PNG" width="50%">
</p>
<p align="center">
  <img src="./images/lesson-5/edge-detection.PNG" width="50%">
</p>
<p align="center">
  <img src="./images/lesson-5/convolution-formula.PNG" width="50%">
</p>
<p align="center">
  <img src="./images/lesson-5/convolution.PNG" width="50%">
</p>

* Edge Handling
  * __Extend__  Corner pixels are extended in 90° wedges. Other edge pixels are extended in lines.
  * __Padding__ The image is padded with a border of 0's, black pixels.
  * __Crop__ Any pixel in the output image which would require values from beyond the edge is skipped.


#### OpenCV & Creating Custom Filters
* [OpenCV](http://opencv.org/about.html) is a computer vision and machine learning software library that includes many common image analysis algorithms that will help us build custom, intelligent computer vision applications.

#### Convolutional Layer
A layer of a deep neural network in which a convolutional filter passes along an input matrix. For example, consider the following 3x3 convolutional filter:

<p align="center">
  <img src="./images/lesson-5/3x3.svg" width="25%">
</p>

The following animation shows a convolutional layer consisting of 9 convolutional operations involving the 5x5 input matrix. Notice that each convolutional operation works on a different 3x3 slice of the input matrix. The resulting 3x3 matrix (on the right) consists of the results of the 9 convolutional operations:

<p align="center">
  <img src="./images/lesson-5/conv-anim.gif" width="50%">
</p>

<p align="center">
  <img src="./images/lesson-5/conv-layer-1.png" width="50%">
</p>

<p align="center">
  <img src="./images/lesson-5/conv-layer-2.png" width="50%">
</p>

* convolutional neural network

  A neural network in which at least one layer is a convolutional layer. A typical convolutional neural network consists of some combination of the following layers:

  * convolutional layers
  * pooling layers
  * dense layers

  Convolutional neural networks have had great success in certain kinds of problems, such as image recognition.

<p align="center">
  <img src="./images/lesson-5/cnn.png" width="50%">
</p>

* See Also:
  * [Convolution](https://developers.google.com/machine-learning/glossary/#convolution)
  * [Convolutional Filter](https://developers.google.com/machine-learning/glossary/#convolutional_filter)
  * [Convolutional Operation](https://developers.google.com/machine-learning/glossary/#convolutional_operation)

#### Convolutional Layers (Part 2)
* Grayscale image -> 2D Matrix
* Color image -> 3 layers of 2D Matrix, one for each channel (Red, Green, Blue)

<p align="center">
  <img src="./images/lesson-5/conv-layer-rgb.PNG" width="50%">
</p>

#### Stride and Padding
* Increase __the number of node__ in convolutional layer -> increase __the number of filter__ 
* increase __the size of detected pattern__ -> increase __the size of filter__
* __Stride__ is the amount by which the filter slides over the image
* Size of convolutional layer depend on what we do at the edge of our image

<p align="center">
  <img src="./images/lesson-5/edge-skip.PNG" width="50%">
</p>

* __Padding__ give filter more space to move by padding zeros to the edge of image

<p align="center">
  <img src="./images/lesson-5/padding.PNG" width="50%">
</p>

#### Pooling Layers
* pooling
  
  Reducing a matrix (or matrices) created by an earlier convolutional layer to a smaller matrix. Pooling usually involves taking either the maximum or average value across the pooled area. For example, suppose we have the following 3x3 matrix:

  <p align="center">
    <img src="./images/lesson-5/PoolingStart.svg" width="25%">
  </p>

  A pooling operation, just like a convolutional operation, divides that matrix into slices and then slides that convolutional operation by strides. For example, suppose the pooling operation divides the convolutional matrix into 2x2 slices with a 1x1 stride. As the following diagram illustrates, four pooling operations take place. Imagine that each pooling operation picks the maximum value of the four in that slice:

  <p align="center">
    <img src="./images/lesson-5/PoolingConvolution.svg" width="75%">
  </p>

  Pooling helps enforce translational invariance in the input matrix.

  Pooling for vision applications is known more formally as spatial pooling. Time-series applications usually refer to pooling as temporal pooling. Less formally, pooling is often called subsampling or downsampling.

#### Increasing Depth
* Incresing depth is actually:
  * extracting more and more complex pattern and features that help identify the content and the objects in an image
  * discarding some spatial information abaout feature like a smooth background that don't help identify the image

<p align="center">
  <img src="./images/lesson-5/increasing-depth.PNG" width="50%">
</p>

#### CNNs for Image Classification

<p align="center">
  <img src="./images/lesson-5/cnn-img-class.PNG" width="50%">
</p>

<p align="center">
  <img src="./images/lesson-5/cnn-img-class-2.PNG" width="50%">
</p>

#### Convolutional Layers in PyTorch
* init

```
self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0)
```

* forward

```
x = F.relu(self.conv1(x))
```

* arguments
  * `in_channels` - number of inputs (in depth)
  * `out_channels` - number of output channels
  * `kernel_size` - height and width (square) of convolutional kernel
  * `stride` - default `1`
  * `padding` - default `0`
  * [documentation](https://pytorch.org/docs/stable/nn.html#conv2d)

* pooling layers

  down sampling factors

  ```
  self.pool = nn.MaxPool2d(2,2)
  ```

  * forward

  ```
  x = F.relu(self.conv1(x))
  x = self.pool(x)
  ```

  * example #1

  ```
  self.conv1 = nn.Conv2d(1, 16, 2, stride=2)
  ```

    * grayscale images (1 depth)
    * 16 filter
    * filter size 2x2
    * filter jump 2 pixels at a time

  * example #2  

  ```
  self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
  ```

    * 16 input from output of example #1
    * 32 filters
    * filter size 3x3
    * jump 1 pixel at a time

* sequential models
    
  ```
  def __init__(self):
        super(ModelName, self).__init__()
        self.features = nn.Sequential(
              nn.Conv2d(1, 16, 2, stride=2),
              nn.MaxPool2d(2, 2),
              nn.ReLU(True),

              nn.Conv2d(16, 32, 3, padding=1),
              nn.MaxPool2d(2, 2),
              nn.ReLU(True) 
         )
  ```

  * formula: number of parameters in a convolutional layer

    * `K` - number of filter
    * `F` - filter size
    * `D_in` - last value in the `input shape`
    
    `(K * F*F * D_in) + K`

  * formula: shape of a convolutional layer

    * `K` - number of filter
    * `F` - filter size
    * `S` - stride
    * `P` - padding
    * `W_in` - size of prev layer

    `((W_in - F + 2P) / S) + 1`

* flattening

  to make all parameters can be seen (as a vector) by a linear classification layer

#### Feature Vector
* a representation that encodes only the content of the image
* often called a feature level representation of an image

<p align="center">
  <img src="./images/lesson-5/feature-vector.PNG" width="50%">
</p>

#### CIFAR Classification Example
* CIFAR-10 (Canadian Institute For Advanced Research) is a popular dataset of 60,000 tiny images

#### Image Augmentation
* data augmentation

  Artificially boosting the range and number of training examples by transforming existing examples to create additional examples. For example, suppose images are one of your features, but your data set doesn't contain enough image examples for the model to learn useful associations. Ideally, you'd add enough labeled images to your data set to enable your model to train properly. If that's not possible, data augmentation can rotate, stretch, and reflect each image to produce many variants of the original picture, possibly yielding enough labeled data to enable excellent training.

<p align="center">
  <img src="./images/lesson-5/image-augmentation.PNG" width="50%">
</p>

* translational invariance
 
  In an image classification problem, an algorithm's ability to successfully classify images even when the position of objects within the image changes. For example, the algorithm can still identify a dog, whether it is in the center of the frame or at the left end of the frame.

* size invariance
 
  In an image classification problem, an algorithm's ability to successfully classify images even when the size of the image changes. For example, the algorithm can still identify a cat whether it consumes 2M pixels or 200K pixels. Note that even the best image classification algorithms still have practical limits on size invariance. For example, an algorithm (or human) is unlikely to correctly classify a cat image consuming only 20 pixels.

* rotational invariance

  In an image classification problem, an algorithm's ability to successfully classify images even when the orientation of the image changes. For example, the algorithm can still identify a tennis racket whether it is pointing up, sideways, or down. Note that rotational invariance is not always desirable; for example, an upside-down 9 should not be classified as a 9.

#### Groundbreaking CNN Architectures
* Since 2010, ImageNet project has held the ImageNet Large Scale Visual Recognition Competition, annual competition for the best CNN for object recognition and classification
* First breakthrough was in 2012, the network called AlexNet was developed by a team at the University of Toronto, they pioneered the use of the ReLU activation function and dropout as a technicque for avoiding overfitting

<p align="center">
  <img src="./images/lesson-5/alexnet.PNG" width="50%">
</p>

* 2014 winner was VGGNet often reffered to as just VGG (Visual Geometry Group) at Oxford University, has two version VGG 16 and VGG 19

<p align="center">
  <img src="./images/lesson-5/vgg.PNG" width="50%">
</p>


* 2015 winner was Microsoft Research called ResNet, like VGG, largest groundbreaking has 152 layers, can solve vanishing gradient problem, achieves superhuman performances in classifying images in ImageNet database

<p align="center">
  <img src="./images/lesson-5/resnet.PNG" width="50%">
</p>

#### Visualizing CNNs (Part 1)
* visualizing the activation maps and convolutional layers
* taking filter from convolutional layers and constructing images that maximize their activations, google researchers get creative with this and designed technique called deep dreams
  * say we have picture of tree, investigate filter for detecting a building, end up creating image that looks like some sort of tree or building hybrid

<p align="center">
  <img src="./images/lesson-5/viz-cnn-1.PNG" width="50%">
</p>

#### Visualizing CNNs (Part 2)
* based on [paper](https://arxiv.org/pdf/1311.2901) by Zeiler and Fergus, visualization using [this toolbox](https://www.youtube.com/watch?v=ghEmQSxT6tw).
  * Layer 1 - pick out very simple shapes and patterns like lines and blobs
  * Layer 2 - circle, stripes and rectangle
  * Layer 3 - complex combinations of features from the second layer
  * Layer 4 - continue progression
  * Layer 5 - classification

<p align="center">
  <img src="./images/lesson-5/viz-cnn-2a.PNG" width="50%">
</p>

<p align="center">
  <img src="./images/lesson-5/viz-cnn-2b.PNG" width="50%">
</p>

<p align="center">
  <img src="./images/lesson-5/viz-cnn-2c.PNG" width="50%">
</p>

#### Summary of CNNs
* take input image then puts image through several convolutional and pooling layers
* result is a set of feature maps reduced in size from the original image
* flatten these maps, creating feature vector that can be passed to series of fully connected linear layer to produce probability distribution of class course
* from thes predicted class label can be extracted
* CNN not restricted to the image calssification task, can be applied to any task with a fixed number of outputs such as regression tasks that look at points on a face or detect human poses

### Quizes
#### Q1 - 5.5: How Computers Interpret Images
* Q: In the case of our 28x28 images, how many entries will the corresponding, image vector have when this matrix is flattened?
* A: `784`
* E: `28*28*1 values = 784`

#### Q2 - 5.6: MLP Structure & Class Scores
<p align="center">
  <img src="./images/lesson-5/q2.PNG" width="50%">
</p>

* Q: After looking at existing work, how many hidden layers will you use in your MLP for image classification?
* A: 2
* E: There is not one correct answer here, but one or two hidden layers should work fine for this simple task, and it's always good to do your research!

#### Q3 - 5.24: Kernels
<p align="center">
  <img src="./images/lesson-5/q3.png" width="50%">
</p>

* Q: Of the four kernels pictured above, which would be best for finding and enhancing horizontal edges and lines in an image?
* A: `d`
* E: This kernel finds the difference between the top and bottom edges surrounding a given pixel.

#### Q4 - 5.32: CNN's for Image Classification
* Q: How might you define a Maxpooling layer, such that it down-samples an input by a factor of 4? 
* A: `nn.MaxPool2d(2,4)`, `nn.MaxPool2d(4,4)`
* E: The best choice would be to use a kernel and stride of 4, so that the maxpooling function sees every input pixel once, but any layer with a stride of 4 will down-sample an input by that factor.

#### Q5 - 5.33: Convolutional Layers in PyTorch

or the following quiz questions, consider an input image that is 130x130 (x, y) and 3 in depth (RGB). Say, this image goes through the following layers in order:

```
nn.Conv2d(3, 10, 3)
nn.MaxPool2d(4, 4)
nn.Conv2d(10, 20, 5, padding=2)
nn.MaxPool2d(2, 2)
```

* Q: After going through all four of these layers in sequence, what is the depth of the final output?
* A: `20`
* E: the final depth is determined by the last convolutional layer, which has a `depth` = `out_channels` = 20.


* Q: What is the x-y size of the output of the final maxpooling layer? Careful to look at how the 130x130 image passes through (and shrinks) as it moved through each convolutional and pooling layer.
* A: 16
* E: The 130x130 image shrinks by one after the first convolutional layer, then is down-sampled by 4 then 2 after each successive maxpooling layer!
  `((W_in - F + 2P) / S) + 1`

  ```
  ((130 - 3 + 2*0) / 1) + 1 = 128
  128 / 4 = 32
  ((32 - 5 + 2*2) / 1) + 1 = 32
  32 / 2 = 16
  ```


* Q: How many parameters, total, will be left after an image passes through all four of the above layers in sequence?
* A: `16*16*20`
* E: It's the x-y size of the final output times the number of final channels/depth = `16*16 * 20`.


### Notebooks
* [Multi-Layer Perceptron, MNIST](https://github.com/agungsantoso/deep-learning-v2-pytorch/blob/master/convolutional-neural-networks/mnist-mlp/mnist_mlp_exercise.ipynb)
* [Multi-Layer Perceptron, MNIST (With Validation)](https://colab.research.google.com/drive/1u4FmtGa24clNIp3sdltqRxyaEHEi-fGe)
* [Creating a Filter, Edge Detection](https://github.com/agungsantoso/deep-learning-v2-pytorch/blob/master/convolutional-neural-networks/conv-visualization/custom_filters.ipynb)
* [Convolutional Layer](https://github.com/agungsantoso/deep-learning-v2-pytorch/blob/master/convolutional-neural-networks/conv-visualization/conv_visualization.ipynb)
* [Maxpooling Layer](https://github.com/agungsantoso/deep-learning-v2-pytorch/blob/master/convolutional-neural-networks/conv-visualization/maxpooling_visualization.ipynb)
* [Convolutional Neural Networks](https://github.com/agungsantoso/deep-learning-v2-pytorch/blob/master/convolutional-neural-networks/cifar-cnn/cifar10_cnn_exercise.ipynb)
* [Convolutional Neural Networks - Image Augmentation](https://github.com/agungsantoso/deep-learning-v2-pytorch/blob/master/convolutional-neural-networks/cifar-cnn/cifar10_cnn_augmentation.ipynb)
