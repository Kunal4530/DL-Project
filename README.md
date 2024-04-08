## Project Title : Plant Disease Prediction

## Problem Statement - 
The challenge is to develop a robust learning model capable of accurately diagnosing infectious diseases in crop plants using image data. With over 50,000 expertly curated images of both healthy and infected crop leaves available on the PlantVillage platform, the goal is to leverage this dataset to train a mobile disease diagnostics system. The model needs to efficiently classify images, enabling farmers to identify and address diseases early, thereby mitigating yield losses caused by infectious pathogens. The objective is to harness the power of crowdsourcing and computer vision to create a scalable solution that can be widely adopted by agricultural communities worldwide.

## Project Explanation:
In this project, I aimed to develop a machine learning model for diagnosing diseases in crop plants, specifically focusing on apple plants. Given the massive dataset obtained from Kaggle, which included images of various crop plant diseases, I decided to narrow down my focus to the apple plant, which had four distinct disease classes.

## Data Preprocessing:
Since the dataset was preprocessed, I proceeded to divide it into training and validation sets. Due to computational limitations, I opted to work with a subset of the data, resulting in 2537 images for training and 634 images for validation.

## Basic Sequential CNN Model (Model 1):
To begin, I constructed a basic sequential convolutional neural network (CNN) comprising seven layers. After compiling the model, I trained it on the training dataset for five epochs, achieving an accuracy of approximately 93%.

## Transfer Learning with VGG16 (Model 2):
Next, I leveraged transfer learning with the VGG16 model from TensorFlow Keras applications. I froze the pre-trained layers and appended custom dense layers. After compiling the model with a custom learning rate, I trained it on the dataset. The resulting accuracy for this model was 96.35%, showcasing the effectiveness of transfer learning.

## Transfer Learning with InceptionV3 (Model 3):
For the third model, I imported InceptionV3 from TensorFlow Keras applications. Similar to the previous model, I froze the pre-trained layers and added custom layers. After compiling the model with a custom learning rate, I trained it on the dataset, achieving an accuracy of around 95%.

## Project Conclusion:
Overall, this project demonstrates the application of machine learning techniques in agricultural contexts, specifically in diagnosing diseases in crop plants. By utilizing transfer learning with pre-trained models like VGG16 and InceptionV3, we significantly improved accuracy while mitigating computational costs. These models hold promise in aiding farmers worldwide in identifying and addressing crop diseases, ultimately contributing to increased food security.

## Dataset Link -
https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset
