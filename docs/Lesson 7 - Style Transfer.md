# 7 - Style Transfer
### Lectures

#### Style Transfer
* apply the style of one image to another image

<p align="center">
  <img src="./images/lesson-7/style-transfer.PNG" width="50%">
</p>

#### Separating Style & Content
* feature space designed to capture texture and color information used, essentially looks at spatial correlations within a layer of a network
* correlation is a measure of the relationship between two or more variables

<p align="center">
  <img src="./images/lesson-7/separate-sytle-content.PNG" width="50%">
</p>

* similarities and differences between features in a layer should give some information about texture and color information found in an image, but at the same time leave out information about the actual arrangement and identitity of different objects in that image

<p align="center">
  <img src="./images/lesson-7/style-representation.PNG" width="50%">
</p>

#### VGG19 & Content Loss
* VGG19 -> 19 layer VGG network

<p align="center">
  <img src="./images/lesson-7/vgg-19.PNG" width="50%">
</p>

* When the network sees the __content image__, it will go through feed-forward process until it gets to a conv layer that is deep in the network, the output will be the content representation

<p align="center">
  <img src="./images/lesson-7/content-rep.PNG" width="50%">
</p>

* When it sees tye __style image__, it will extract different features from multiple layers that represent the style of that image

<p align="center">
  <img src="./images/lesson-7/style-rep.PNG" width="50%">
</p>

* __content loss__ is a loss that calculates the difference between the content (Cc) and target (Tc) image representation

<p align="center">
  <img src="./images/lesson-7/content-loss.PNG" width="50%">
</p>

#### Gram Matrix
* Correlations at each layer in convolutional layer are given by a Gram matrix
* First step in calculating the Gram matrix, will be to vectorize the values of feature map

<p align="center">
  <img src="./images/lesson-7/flatten.PNG" width="50%">
</p>

* By flattening the XY dimensions of the feature maps, we're convrting a 3D conv layer to a 2D matrix of values

<p align="center">
  <img src="./images/lesson-7/vectorized-feature-map.PNG" width="50%">
</p>

* The next step is to multiply vectorized feature map by its transpose to get the gram matrix

<p align="center">
  <img src="./images/lesson-7/gram-matrix.PNG" width="50%">
</p>

#### Style Loss

* __content loss__ is a loss that calculates the difference between the image style (Ss) and target (Ts) image style, `a` is constant that accounts for the number of values in each layer, `w` is style weights

<p align="center">
  <img src="./images/lesson-7/style-loss.PNG" width="50%">
</p>

* Add together content loss and style loss to get total loss and then use typical back propagation and optimization to reduce total loss

<p align="center">
  <img src="./images/lesson-7/total-loss.PNG" width="50%">
</p>

#### Loss Weights

* alpha beta ratio is ratio between alpha (content weight) and beta (style weight)

<p align="center">
  <img src="./images/lesson-7/weight-ratio.PNG" width="50%">
</p>

* Different alpha beta ratio can result in different generated image

<p align="center">
  <img src="./images/lesson-7/weight-ratio-effect.PNG" width="50%">
</p>

### Quizes
#### Q1 - 6.4: Gram Matrix
##### Q 1.1
* Q: Given a convolutional layer with dimensions `d x h x w = (20*8*8)`, what length will one row of the vectorized convolutional layer have? (Vectorized means that the spatial dimensions are flattened.)
* A: `64`
* E: When the height and width (8 x 8) are flattened, the resultant 2D matrix will have as many columns as the height and width, multiplied: `8*8 = 64`.

##### Q 1.2
* Q: Given a convolutional layer with dimensions `d x h x w = (20*8*8)`, what dimensions (h x w) will the resultant Gram matrix have?
* A: `(20 x 20)`
* E: The Gram matrix will be a square matrix, with a width and height = to the depth of the convolutional layer in question.

### Notebooks
* [Style Transfer with Deep Neural Networks](https://github.com/agungsantoso/deep-learning-v2-pytorch/blob/master/style-transfer/Style_Transfer_Exercise.ipynb)
