
# Chest X-Ray Image Classification

This repository contains code for a
chest X-ray image classification project. The goal is to classify X-ray images into two classes: \"NORMAL\" and \"PNEUMONIA\". The project involves **data** **preprocessing**, **unsupervised** **learning** using **KMeans** and **GMM**, and **supervised learning** using Convolutional Neural Networks (**CNNs**).

Below are the steps to set up the project and run the code.

### Library Installation 

Make sure to install the required libraries by
running the following commands:

 ```{python}

pip install torch torchvision opencv-contrib-python
pip install scikit-learn kaggle 
```

### Downloading the Data 

Before running the code, download the chest X-ray dataset from Kaggle. Ensure you have a Kaggle account and have obtained the API key. Move the kaggle.json file to the required directory:

 ```{python}
mkdir -p \~/.kaggle cp ./kaggle.json \~/.kaggle/ chmod 600
\~/.kaggle/kaggle.json 
```

Download and unzip the dataset:
```
kaggle datasets download -d paultimothymooney/chest-xray-pneumonia 
unzip chest-xray-pneumonia.zip 
 ```
### Data Preprocessing 
The code includes data loading, **histogram equalization** resizing, and normalization. The dataset is split into training, testing, and validation sets.

### Visualization 
Explore the dataset through visualization, including
randomly selected X-ray images and checking for class imbalance.

### Data Augmentation 
To address class imbalance, data augmentation techniques are applied, including random resizing, flipping, Gaussian blur, and affine transformations.

### Unsupervised Learning
Apply unsupervised learning techniques such as KMeans and Gaussian Mixture Models (GMM) for feature extraction and clustering.

### Supervised Learning

Train a Convolutional Neural Network (CNN) for image classification. Two architectures are implemented: a custom CNN and a fine-tuned MobileNetV3.

### Model Evaluation 

Evaluate the models using metrics such as accuracy, precision, recall, and F1-score. Confusion matrices and ROC curves are also plotted for further analysis.

##Comparison 
We compared the performance of unsupervised learning (KMeans and GMM) with supervised learning (CNN) and discussed their strengths and weaknesses.
