# ImageClustering
<p align='center'>Made with :heart: by <b>Amr Elzawawy</b></p>

![](https://sds-platform-private.s3-us-east-2.amazonaws.com/uploads/64_blog_image_2.png)

This work was developed in assignment 3 for *AI Course Fall 2019/2020 offering at AlexU Faculty of Engineering*. In this assignment I implemented K-Means clustering algorithm from scratch and applied it on an image dataset with different experiment runs.

The **purpose** of K-means is to identify groups, or clusters of data points in a multidimensional space. The number K in K-means is the **number of clusters** to create. Initial cluster means are usually chosen **at random.**
K-means is usually implemented as an **iterative procedure** in which each iteration involves two successive steps. 

- The first step is to assign each of the data points to a cluster. 
- The second step is to modify the cluster means so that they become the mean of all the points assigned to that cluster.

The quality of the current assignment is given by the distortion measure which is the sum of squared distances between each cluster centroid and points inside the cluster.

## The CIFAR-10 dataset
The CIFAR-10 dataset consists of **60,000** 32x32 colour images in 10 classes, with 6000 images per class. There are 50,000 training images and 10,000 test images.

The dataset is divided into **five training batches** and **one test batch**, each with 10,000 images. The test batch contains exactly 1,000 randomly-selected images from each class. The training batches contain the remaining images in random order, but some training batches may contain more images from one class than another. Between them, the training batches contain exactly 5,000 images from each class.

You can check and download the dataset from [here](https://www.cs.toronto.edu/~kriz/cifar.html)

## Loading Dataset Into Memory

Wrote my **own ImageDataLoader Class** that is extendible to support multiple datasets in the future as well.
It loads the dataset using the directory path, and returns a tuple of four results: **train_X, train_y , test_X , test_y** 

Although clustering does not make use of the labels since this is an unsupervised learning algorithm.
This is intended to support a general case ImageDataLoader not specifically built for the image clustering problem. 

### Final notes
- I open-source my KMeans Implementation, and CIFAR-10 Data Loader class Implementation.
- Intend to add my Spectral Clustering Implementation later on.
- Intend to make my Data Loader class usable for more than one dataset.
