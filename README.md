# This is the second deep learning project. This time, I will classify 5 different flowers.

  I will train a model with total 100 epochs first. (50 for training-50 for tuning) Then, I  will observe the plot figure of validation loss and try to decide for the optimum epoch value to prevent overfitting problem. Finally, I will evaluate the results that I gained and decide which one is better.
  
  The tools will be used during this project are Anaconda to create environment, jupyter and spyder to code, some libraries such as TensorFlow, Keras, Matplotlib, Seaborn, Pandas, Numpy, Sklearn.
  
# Differences Between Binary Classification Project and Categorical Classification Project:

* I will use 3 different deep learning models. VGG16, MobileNetV2 and ResNet50.
* The optimization algorithm will be ADAM as I got the best results from the last project which I did for binary classification.
* At first, Input shape equals to (224, 224, 3). With GlobalAveragePooling2D module, the shape will become 2D. At the binary classification, we used Flatten to get 224 * 224 * 3 = 150528 neurons for input layers.
* We will use 512 neurons for hidden layers to train a deep learning model with theirs activation function equals to relu.
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
I run the code.

```python
-python modelBuilding_multiclass.py
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


**So, the accuracy is 0.87. It is not that bad for multiclass classification. But the loss is a bit much. Let us try if we can decrease the loss and improve the model for better classification. First, last epochs of rerun model.**


![last 5 epochs](https://user-images.githubusercontent.com/44292203/118889979-3a0bfc00-b8fe-11eb-93f6-532886478063.PNG)


Now, loss figure.


![loss graph](https://user-images.githubusercontent.com/44292203/118890017-43956400-b8fe-11eb-842d-3955ec54b36c.PNG)


Running the testing code.


```python
-python modelTesting_multiclass.py
```


So, let us check the results. First, the confusion matrix. 16 tulips are predicted as rose and 13 dandelions are predicted as daisies. Accuracy of daisy is very bad when we compare with the others.


![conf mat](https://user-images.githubusercontent.com/44292203/118890055-5314ad00-b8fe-11eb-9485-4fb23711e5f5.PNG)


Finally, total loss and total accuracy for testing.


![total acc and loss](https://user-images.githubusercontent.com/44292203/118890103-66c01380-b8fe-11eb-91dc-ed736e6c7824.PNG)


**So, the accuracy decreased a bit, but it is not necessary. Difference is not significantlyfor accuracy. But when we check the loss, the loss improved with decreasing by almost %16.**

# Results for MobileNetV2 Model

Now, I have changed the model function to use MobileNetV2 architecture and count of top layers to be unfrozen. Comparing with VGG16 model, this model took less time.


![model](https://user-images.githubusercontent.com/44292203/118891847-44c79080-b900-11eb-9d1e-49a932bfdaef.PNG)


135 top layers will be unfrozen.


![layer count](https://user-images.githubusercontent.com/44292203/118891873-4ee98f00-b900-11eb-8b4f-8e81dc0803db.PNG)


I began to train the model. This is the first value of validation loss and validation accuracy at the beginning. Comparing with others, this model has the highest validation accuracy and the lowest validation loss at the beginning.


![first epoch](https://user-images.githubusercontent.com/44292203/118891977-7b9da680-b900-11eb-8c68-bb9eac830c8f.PNG)


These are the last 5 epochs for training the model. 


![last 5 for tra](https://user-images.githubusercontent.com/44292203/118892027-8ce6b300-b900-11eb-9027-17db760efae5.PNG)


These are the last 5 epochs for tuning.


![last 5 for tun](https://user-images.githubusercontent.com/44292203/118892040-93752a80-b900-11eb-9b66-b2f40cb29107.PNG)


Accuracy and loss figures.


![acc figure](https://user-images.githubusercontent.com/44292203/118892059-9bcd6580-b900-11eb-8bc5-dcd9e5a554c0.PNG)


![loss figure](https://user-images.githubusercontent.com/44292203/118892072-9f60ec80-b900-11eb-9cf0-c77af34b2168.PNG)


I see the lowest epoch at 72. epoch. But after I rerun the model with 72 epochs, I got worse result than I got at the beginning. I rerun the model with at least 6 different epoch numbers, and I got the best result with 69 epochs. But the result is not good enough. These are the confusion matrix and total accuracy and loss for testing.


![conf mat](https://user-images.githubusercontent.com/44292203/118892135-bb648e00-b900-11eb-9dac-ebf805c65b73.PNG)


Prediction for rose and daisy is not good enough. Machine predicted 24 roses as tulips, 20 daisies as dandelions. But the prediction for dandelion is very good.


![total test](https://user-images.githubusercontent.com/44292203/118892173-c9b2aa00-b900-11eb-8d7d-538a4a8a2dba.PNG)


**I got the accuracy as 0.86 and loss as 0.45. Comparing with VGG16, accuracy decreased a little, but I have a good improvement for loss. Let us compare the results with 69 epochs model. First, last 5 epochs for tuning.**


![last 5 for tuning](https://user-images.githubusercontent.com/44292203/118892237-dfc06a80-b900-11eb-9583-d6f34d2719a5.PNG)


The loss figure.


![loss figure](https://user-images.githubusercontent.com/44292203/118892256-e818a580-b900-11eb-9756-d3ab750a669d.PNG)


Now confusion matrix and total accuracy and total loss for testing.


![conf](https://user-images.githubusercontent.com/44292203/118892278-f1a20d80-b900-11eb-9fdb-b46c11870a2c.PNG)


Prediction of tulip has been improved but prediction daisy is still bad. Prediction of dandelion has decreased a little and prediction of rose and sunflower has increased.


![total](https://user-images.githubusercontent.com/44292203/118892286-f49cfe00-b900-11eb-8eb0-f9f498c25d15.PNG)


**Accuracy has decreased a little, but the loss has improved as about %5. This is what I could get the most improvement.**


# Results for ResNet50 Model

I have changed the model function in the code and number of top layers to be unfrozen.


![new model func](https://user-images.githubusercontent.com/44292203/118892619-7856ea80-b901-11eb-935b-e86aa9c7695f.PNG)


154 top layers will be unfrozen.


![unfrozen count](https://user-images.githubusercontent.com/44292203/118892627-7b51db00-b901-11eb-908e-16ad0fdd52df.PNG)


I began training the model. This is the value of validation loss and validation accuracyat the beginning.


![first epochs](https://user-images.githubusercontent.com/44292203/118892720-9cb2c700-b901-11eb-980d-84ecae3877f9.PNG)


Now, the last 5 epochs for training.


![last 5 for tra](https://user-images.githubusercontent.com/44292203/118892744-a5a39880-b901-11eb-8d36-b1446738860e.PNG)


And the last 5 epochs for tuning.


![last 5 for tun](https://user-images.githubusercontent.com/44292203/118892773-b05e2d80-b901-11eb-818c-58a777d16fda.PNG)


Accuracy and loss figures.


![acc fig](https://user-images.githubusercontent.com/44292203/118892805-b9e79580-b901-11eb-971e-2575a1e21f96.PNG)


![57-74](https://user-images.githubusercontent.com/44292203/118892813-be13b300-b901-11eb-8b78-8bad1e11b3ec.PNG)


It seems like I got the lowest validation loss in 57. epoch. Let use build a new model with 57 epochs and compare them. But first, the results of the model with 100 epochs.


![conf mat](https://user-images.githubusercontent.com/44292203/118892860-d71c6400-b901-11eb-9002-d7c22772ec6b.PNG)


Prediction of dandelion is very good at this model, but the prediction of daisy is not good enough again.


![total test and acc](https://user-images.githubusercontent.com/44292203/118892884-e26f8f80-b901-11eb-8058-bd1b57279d32.PNG)


**Accuracy is the best so far when I compare with other models, but the loss is very bad. I will build another model with the same architecture and see if I can decrease it or not. First, the last epochs of the new model.**


![last epochs for tuning](https://user-images.githubusercontent.com/44292203/118892947-f74c2300-b901-11eb-9770-f045992dfe0c.PNG)


Now, the loss figure.


![loss fig](https://user-images.githubusercontent.com/44292203/118892999-0c28b680-b902-11eb-95b7-2ce824f07138.PNG)


And now the results. This is the confusion matrix.


![conf matr](https://user-images.githubusercontent.com/44292203/118893029-15b21e80-b902-11eb-9c0a-fd8c18c8e3c3.PNG)


And now, total loss and accuracy for testing.


![total test and acc](https://user-images.githubusercontent.com/44292203/118893062-22cf0d80-b902-11eb-9b51-0f6440eaf2cf.PNG)


**So, I managed to decrease the loss by almost %30. There is not a big loss for accuracy. So, I managed to improve my model.**

# Result Analysis and Summary

I have 3 different architectures and 6 different model with Adam optimizer and 
different epoch numbers.


1-) VGG16 Model with 100 epochs approximately:

* Accuracy: 0.87
* Loss: 0.64


2-) VGG16 Model with 86 epochs approximately:

* Accuracy: 0.87
* Loss: 0.54


3-) MobileNetV2 Model with 100 epochs approximately:

* Accuracy: 0.86
* Loss: 0.45


4-) MobileNetV2 Model with 69 epochs approximately:

* Accuracy: 0.85
* Loss: 0.43


5-) ResNet50 Model with 100 epochs approximately:

* Accuracy: 0.88
* Loss: 0.61


6-) ResNet50 Model with 57 epochs approximately:

* Accuracy: 0.86
* Loss: 0.43

**So, the most effective model from these 3 different architectures and 6 different models is using ResNet50 Model with 57 epochs. I did not get the highest accuracy in this model but comparing with others, the difference is not significantly but the difference between losses is very significantly. MobileNetV2 model with 69 epochs actually very close to ResNet50 Model with 57 epochs but accuracy is distinctions between the two models. This is what I can do best with my GPU.**


## To see the .h5 models that I've built and images that I've used

* [Images](img/flowers)
* [Models](https://drive.google.com/drive/folders/1abpm0u8zIyytIAYbtpKrKE7ATWsAaVCD?usp=sharing)
