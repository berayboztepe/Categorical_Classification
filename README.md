# This is the second deep learning project. This time, I will classify 5 different flowers.

  I will train a model with total 100 epochs first. (50 for training-50 for tuning) Then, I  will observe the plot figure of validation loss and try to decide for the optimum epoch value to prevent overfitting problem. Finally, I will evaluate the results that I gained and decide which one is better.
  
  The tools will be used during this project are Anaconda to create environment, jupyter and spyder to code, some libraries such as TensorFlow, Keras, Matplotlib, Seaborn, Pandas, Numpy, Sklearn.
  
# Differences Between Binary Classification Project and Categorical Classification Project:

* I will use 3 different deep learning models. VGG16, MobileNetV2 and ResNet50.
* The optimization algorithm will be ADAM as I got the best results from the last project which I did for binary classification.
* At first, Input shape equals to (224, 224, 3). With GlobalAveragePooling2D module, the shape will become 2D. At the binary classification, we used Flatten to get 224*224*3 = 150528 neurons for input layers.
* We will use 512 neurons for hidden layers to train a deep learning model with theirsactivation function equals to relu.
* And we have 5 neurons for output layer which equals to class numbers. Activation function of output layers’ is softmax.
* Also, the difference between binary and categorical classification is class_type equals to categorical and we are using loss function as categorical_crossentropy.

# Characteristics of the data

  There are 600 images for each class. Total size is 150 MB. Images are colourful images which all are representing RGB. There is total 1200 images for training (240 for each class). Also, there are 800 images for test and validation both (180 for each class).
  

![Resim6](https://user-images.githubusercontent.com/44292203/118888863-88200000-b8fc-11eb-8052-22726ef31f3c.png)


![Resim7](https://user-images.githubusercontent.com/44292203/118888872-8c4c1d80-b8fc-11eb-9c36-7da002cf4a37.png)


I have 5 different classes of flowers. These are:
* Daisy


![Resim1](https://user-images.githubusercontent.com/44292203/118888369-db458300-b8fb-11eb-922a-d7dd6af0b135.jpg)


* Dandelion


![Resim2](https://user-images.githubusercontent.com/44292203/118888376-de407380-b8fb-11eb-9e6f-00bffa2ba1b2.jpg)


* Rose


![Resim3](https://user-images.githubusercontent.com/44292203/118888382-e00a3700-b8fb-11eb-8691-dfb66f288ca3.jpg)


* Sunflower


![Resim4](https://user-images.githubusercontent.com/44292203/118888383-e00a3700-b8fb-11eb-931e-58a6eb313433.jpg)


* Tulip


![Resim5](https://user-images.githubusercontent.com/44292203/118888384-e0a2cd80-b8fb-11eb-89db-a9954609eb1f.jpg)



# Python Code

  First, activating the environment.

```python
-conda activate tf-gpu
```
Setting the parameters. Class number will be changed due to number of classes in printClasses function.
```
CLASS_MODE = 'categorical'
LOSS_TYPE ='categorical_crossentropy'    
CLASSES_NUMBER = 0
IMAGE_SIZE = (224, 224)
INPUT_SHAPE = (224, 224, 3)
TRAIN_BATCH_SIZE = 8
VAL_BATCH_SIZE = 8
TRAINING_EPOCHS = 50 
TUNNING_EPOCHS = 50
ACC = VAL_ACC = LOSS = VAL_LOSS = None
```
Optimizitaion function will be the same that I got the best result in the first project.

```python
model.compile(loss=LOSS_TYPE,
                  optimizer=optimizers.Adam(lr=LR),
                  metrics=['acc'])  
```
Code for the first architecture, VGG16
```
from keras.applications import VGG16
baseModel = VGG16(weights='imagenet', include_top=False, input_shape=INPUT_SHAPE)
```
When unfreezing the model, 11 top layers will be unfrozen.
```
unfreezeModel(baseModel, 11)
```
I run the function.

```python
-python modelBuilding.py
```


# Results for VGG16 Model
  
The model training has begun. This is the result of first epoch. Validation loss and accuracy is very low at the beginning.


![first epoch](https://user-images.githubusercontent.com/44292203/118889703-c79b1c00-b8fd-11eb-8510-d68c75a897be.PNG)


Last 5 epochs for training.


![last 5 for training](https://user-images.githubusercontent.com/44292203/118889720-d08bed80-b8fd-11eb-934b-786964b405c8.PNG)


Last 5 epochs for tuning.


![last 5 epochs for tuning](https://user-images.githubusercontent.com/44292203/118889760-de417300-b8fd-11eb-903e-a220a64172a7.PNG)


As can be seen, validation accuracy has increased to almost 0.86 from 0.24. There is accuracy figure.


![acc figure](https://user-images.githubusercontent.com/44292203/118889807-f0bbac80-b8fd-11eb-9abe-7fd2c273e2fd.PNG)


And the loss figure. Validation loss is changing so much but I got the lowest value of validation loss in the 86. epoch. So, I will rerun the code and see if we improve our model or not.


![loss figure](https://user-images.githubusercontent.com/44292203/118889872-11840200-b8fe-11eb-94c1-91d559c76d8e.PNG)



And these are the results of testing. First Confusion Matrix. As can be seen, there are some mistakes. For example, 18 daisies are predicted as dandelion and 10 sunflowers are predicted as dandelion. We have total 112 mistakes from 900 images of data.


![conf matrix](https://user-images.githubusercontent.com/44292203/118889905-1cd72d80-b8fe-11eb-82dc-860c6bb43c09.PNG)


And let us see the accuracy and loss for testing.


![total loss and acc](https://user-images.githubusercontent.com/44292203/118889936-2b254980-b8fe-11eb-8d99-6a2dea970165.PNG)


So, the accuracy is 0.87. It is not that bad for multiclass classification. But the loss is a bit much. Let us try if we can decrease the loss and improve the model for better classification. First, last epochs of rerun model.


![last 5 epochs](https://user-images.githubusercontent.com/44292203/118889979-3a0bfc00-b8fe-11eb-93f6-532886478063.PNG)


Now, loss figure.


![loss graph](https://user-images.githubusercontent.com/44292203/118890017-43956400-b8fe-11eb-842d-3955ec54b36c.PNG)


So, let us check the results. First, the confusion matrix. 16 tulips are predicted as rose and 13 dandelions are predicted as daisies. Accuracy of daisy is very bad when we compare with the others.


![conf mat](https://user-images.githubusercontent.com/44292203/118890055-5314ad00-b8fe-11eb-9485-4fb23711e5f5.PNG)


Finally, total loss and total accuracy for testing.


![total acc and loss](https://user-images.githubusercontent.com/44292203/118890103-66c01380-b8fe-11eb-91dc-ed736e6c7824.PNG)


**So, the accuracy decreased a bit, but it is not necessary. Difference is not significantlyfor accuracy. But when we check the loss, the loss improved with decreasing by almost %16.**

# Results for MobileNetV2 Model:

