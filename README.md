# Automatic classification of skin lesion

Classification of skin lesion images by a machine learning approach.

The goal of this project is to classify images of skin lesion into two classes: malignant tumour (label 0) or benign (label 1) tumour. The images in question in this project are as follows:

<p align="center">
<img  src="https://github.com/LefdRida/Automatic-classification-of-skin-lesion/blob/main/images/0%261%20tumours.JPG" height="400" width="700"/>
</p>

For this purpose wa have applied classification algorithm on a data set where each sample is composed from 3 types of image:
  - Original image: the RGB image of the tumor. 
  - The segmented image: Binary image 
  - Superpixel image
  
<p align="center">
<img  src="https://github.com/LefdRida/Automatic-classification-of-skin-lesion/blob/main/images/sample_of_data.JPG" height="400" />
</p>

Each sample has a label of 0 or 1 in a csv file.

## Features extraction

As we have used classical machine learning models (i.e non deep learning models), we have extracted features from the images at our disposal.
  - from original image, we have extracted texture descriptors by applying LBP algorithm on the image. The output of this algorithm is a vector of dimension (256,1). also we have extracted the number of colors existed in the tumor (TODO: add ref of color table). 
  - from segmented image we have extracted geometrical features such as area, perimeter, diameter (TODO: add other features). from binary image, we have extracted 11 features.
  - from superpixel images we have extracted 1 feature which is the number of superpixel.
 
 ## Models & results

In this project, we have used only machine learning algorithm for binary classification such as :
  - SVM
  - Logistic regression 
  - Decision Tree 
  - Naive baysian model 
  - SGD classifier: this means linear classifier with SGD as optimization algorithm for training
  
The traning has been done with a dataset where every sample has 269 features. The perfomances of the models on the test set are as follows:

<p align="center">
<img  src="https://github.com/LefdRida/Automatic-classification-of-skin-lesion/blob/main/images/performance.JPG" height="400" />
</p>

