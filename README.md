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
  
I have 5 different classes of flowers. These are:
* Daisy
* Dandelion
* Rose
* Sunflower
* Tulip
