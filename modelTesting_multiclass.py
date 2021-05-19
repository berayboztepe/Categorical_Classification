print("\033[2J")
import os
from sklearn.metrics import confusion_matrix, classification_report
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from keras.preprocessing.image import ImageDataGenerator
import itertools

# *** PARAMETERS ***
IMAGE_SIZE = (224, 224)
INPUT_SHAPE = (224, 224, 3)
TRAIN_BATCH_SIZE = 8
VAL_BATCH_SIZE = 8
CLASS_MODE = 'categorical'        

# *** PATHS ***
MODEL_NAME = 'resnet_model_3.h5'
BASE_DATASET_FOLDER = './' + 'flowers'
TEST_FOLDER = os.path.join(BASE_DATASET_FOLDER, 'test')
     
def plotCM(cm, classes, cmap=plt.cm.Blues):    
    plt.figure(figsize=(10, 10))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)        
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, horizontalalignment="center", fontsize=24)
    plt.yticks(tick_marks, classes, rotation=90, verticalalignment="center", fontsize=24)	    
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j]),
                 horizontalalignment="center",
				 verticalalignment="center",
                 color="white" if cm[i, j] > thresh else "black",
                 fontsize=33)    
    plt.ylabel('True class', fontsize=30)
    plt.xlabel('Predicted class', fontsize=30)
    plt.show()

def testGenerator():
    test_datagen = ImageDataGenerator(rescale=1./255)
    test_generator = test_datagen.flow_from_directory(
            TEST_FOLDER,
            target_size=IMAGE_SIZE,                
            batch_size=1,
            class_mode=CLASS_MODE, 
            shuffle=False)
    return test_generator

def main():        
    print("\n*** MODEL TESTING HAS BEGUN ***\n")        
    model = keras.models.load_model('./' + MODEL_NAME)
    test_generator = testGenerator()
    classes = {v: k for k, v in test_generator.class_indices.items()}    
    steps=test_generator.samples//test_generator.batch_size
    test_generator.reset()
    loss, accuracy = model.evaluate_generator(test_generator, steps, verbose=1)
    print("\nAccuracy: %f\nLoss: %f\n" % (accuracy,loss))
    test_generator.reset()    
    Y_pred = model.predict_generator(test_generator, steps, verbose=1)
    y_pred = np.argmax(Y_pred, axis=1)            
    cnf_matrix = confusion_matrix(test_generator.classes, y_pred)        
    plotCM(cnf_matrix, list(classes.values()))        
    print('')
    print(classification_report(test_generator.classes, y_pred, target_names=list(classes.values())))
    labels = (test_generator.class_indices)
    labels = dict((v, k) for k, v in labels.items())
    predictions = [labels[k] for k in y_pred]
    filenames=test_generator.filenames
    results=pd.DataFrame({"Filename":filenames, "Predictions":predictions})
    results.to_csv("results.csv",index=False)

if __name__ == "__main__":
    main()
