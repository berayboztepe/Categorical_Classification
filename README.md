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















