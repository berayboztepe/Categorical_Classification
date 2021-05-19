print("\033[2J")
import os
from glob import iglob
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, GlobalAveragePooling2D
import seaborn as sns

# *** PARAMETERS ***
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

# *** PATHS ***
MODEL_NAME = 'mobile_net_model.h5'
BASE_DATASET_FOLDER = './' + 'flowers'
TRAIN_FOLDER = os.path.join(BASE_DATASET_FOLDER, 'train')
VALIDATION_FOLDER = os.path.join(BASE_DATASET_FOLDER, 'validation')
     
def drawDataSplit():
    plot_dataset_description(TRAIN_FOLDER, "Class sizes for training data")
    plot_dataset_description(VALIDATION_FOLDER, "Class sizes for validation data")

def smooth_curve(points, factor):
    smoothed_points = []
    for point in points:
        if smoothed_points:
            previous = smoothed_points[-1]
            smoothed_points.append(previous * factor + point * (1 - factor))
        else:
            smoothed_points.append(point)
    return smoothed_points

def drawTunningHistoryDiagrams(tunningHistory):
    global ACC, VAL_ACC, LOSS, VAL_LOSS
    sns.set()
    sns.set(color_codes=True)
    sns.set_context("poster", font_scale=1.5, rc={"lines.linewidth": 3})    
    ACC += tunningHistory.history['acc']
    VAL_ACC += tunningHistory.history['val_acc']
    LOSS += tunningHistory.history['loss']
    VAL_LOSS += tunningHistory.history['val_loss']    
    epochs = range(len(ACC))
    
    plt.figure(1, figsize=(10, 8))
    #plt.plot(epochs, ACC, color='blue', label='Training Accuracy')
    #plt.plot(epochs, VAL_ACC, color='red', label='Validation Accuracy')
    plt.plot(epochs,  smooth_curve(ACC, 0.8), color='blue', label='Training Accuracy')
    plt.plot(epochs, smooth_curve(VAL_ACC, 0.8), color='red', label='Validation Accuracy')
    plt.plot([TRAINING_EPOCHS, TRAINING_EPOCHS], plt.ylim(), label='Start Fine Tuning', color='green')    
    plt.legend(loc='lower right')
    plt.xlabel('Epoch', fontsize=40)
    plt.ylabel('Accuracy', fontsize=40)
    plt.xticks(fontsize=40)
    plt.yticks(fontsize=40)
    plt.legend(fontsize=30)
    plt.show()    
    
    plt.figure(2, figsize=(10, 8))
    #plt.plot(epochs, LOSS, color='blue', label='Training Loss')
    #plt.plot(epochs, VAL_LOSS, color='red', label='Validation Loss')
    plt.plot(epochs, smooth_curve(LOSS, 0.8), color='blue', label='Training Loss')
    plt.plot(epochs, smooth_curve(VAL_LOSS, 0.8), color='red', label='Validation Loss')
    plt.plot([TRAINING_EPOCHS, TRAINING_EPOCHS], plt.ylim(), label='Start Fine Tuning', color='green')    
    plt.legend(loc='upper right')    
    plt.xlabel('Epoch', fontsize=40)
    plt.ylabel('Loss', fontsize=40)
    plt.xticks(fontsize=40)
    plt.yticks(fontsize=40)
    plt.legend(fontsize=30)
    plt.show()    

def percentage_value(pct, allvals):
    absolute = int(pct/100.*np.sum(allvals))
    return "{:.1f}%\n({:d})".format(pct, absolute)

def plot_dataset_description(path, title):
    classes = []  
    for filename in iglob(os.path.join(path, "**","*.jpg")):
        classes.append(os.path.split(os.path.split(filename)[0])[-1])
    classes_cnt = Counter(classes)
    values = list(classes_cnt.values())
    labels = list(classes_cnt.keys())        
    plt.figure(figsize=(8, 8))
    plt.rcParams['font.size']=20    
    plt.pie(values, labels=labels, autopct=lambda pct: percentage_value(pct, values), 
            shadow=True, startangle=140)
    plt.title(title)    
    plt.show()

# BUILDING A TRAINING DATA GENERATOR 
def trainGenerator():
    train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=40,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest')    
    train_generator = train_datagen.flow_from_directory(
            TRAIN_FOLDER,
            target_size=IMAGE_SIZE,        
            batch_size=TRAIN_BATCH_SIZE,
            class_mode=CLASS_MODE, 
            shuffle=True)          
    return train_generator

# BUILDING A VALIDATION DATA GENERATOR
def validationGenerator():    
    val_datagen = ImageDataGenerator(rescale=1./255)
    val_generator = val_datagen.flow_from_directory(
            VALIDATION_FOLDER,
            target_size=IMAGE_SIZE,		
            batch_size=VAL_BATCH_SIZE,
            class_mode=CLASS_MODE, 
            shuffle=True)    
    return val_generator

def printClasses(train_generator):    
    global CLASSES_NUMBER
    classes = {v: k for k, v in train_generator.class_indices.items()}
    CLASSES_NUMBER = len(classes)
    print("\n*** CLASSES OF DATA ***")
    print("NUMBER OF CLASSES: " + str(CLASSES_NUMBER))
    print(classes)
    print()    

# FREEZING THE BASE MODEL
def freezeModel(baseModel):    
    for layer in baseModel.layers:
        layer.trainable = False 

# UNFREEZING THE UPPER LAYERS OF THE BASE MODEL
def unfreezeModel(baseModel, layer): 
    for layer in baseModel.layers[layer:]:
        layer.trainable = True

def compileModel(model, LR):
    model.compile(loss=LOSS_TYPE,
                  optimizer=optimizers.Adam(lr=LR),
                  metrics=['acc'])    

def modelTraining(model, train_generator, val_generator):
    print("\n*** NEW LAYERS TRAINING - THE WHOLE BASE MODEL IS FROZEN ***")
    trainingHistory = model.fit_generator(train_generator,
            steps_per_epoch=train_generator.samples//train_generator.batch_size,
            epochs=TRAINING_EPOCHS,
            validation_data=val_generator,
            validation_steps=val_generator.samples//val_generator.batch_size,
            verbose=1)        
    global ACC, VAL_ACC, LOSS, VAL_LOSS        
    ACC = trainingHistory.history['acc']
    VAL_ACC = trainingHistory.history['val_acc']
    LOSS = trainingHistory.history['loss']
    VAL_LOSS = trainingHistory.history['val_loss'] 
    print("\n*** NEW LAYERS TRAINING HAS BEEN COMPLETED ***")
    return trainingHistory
    
def modelTunning(model, trainingHistory, train_generator, val_generator):
    print("\n*** TUNING - THE TOP LAYERS OF THE BASE MODEL ARE UNFROZEN ***")    
    TOTAL_EPOCHS = TRAINING_EPOCHS + TUNNING_EPOCHS    
    tunningHistory = model.fit_generator(
            train_generator,
            steps_per_epoch=train_generator.samples//train_generator.batch_size,
            epochs=TOTAL_EPOCHS,
            initial_epoch = trainingHistory.epoch[-1],
            validation_data=val_generator,
            validation_steps=val_generator.samples//val_generator.batch_size,
            verbose=1)
    print("\n*** MODEL TUNING HAS BEEN COMPLETED ***")
    model.save(MODEL_NAME)    
    return tunningHistory

def modelRN50():
    print("\n*** DOWNLOADING THE BASE MODEL ***")
    from keras.applications import MobileNetV2
    baseModel = MobileNetV2(weights='imagenet', include_top=False, input_shape=INPUT_SHAPE)         
    model = Sequential()    
    model.add(baseModel)
    model.add(GlobalAveragePooling2D())    
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.3))    
    model.add(Dense(CLASSES_NUMBER, activation='softmax'))    
    print("\n*** NEW LAYERS WERE ADDED TO THE BASE MODEL ***")
    return baseModel, model

def main():        
    print("\n*** MODEL BUILDING HAS BEGUN ***")
    
    # DATA GENERATORS BUILDING
    train_generator = trainGenerator()
    val_generator = validationGenerator()        
    printClasses(train_generator)    
    
    drawDataSplit()
    
    # BUILDING A MODEL
    baseModel, model = modelRN50()        
    
    # FREEZING THE BASE MODEL
    freezeModel(baseModel)
    compileModel(model, 2e-5)    
    
    # MODEL TRAINING
    trainingHistory = modelTraining(model, train_generator, val_generator)    
    
    # UNFREEZING THE UPPER LAYERS OF THE BASE MODEL    
    unfreezeModel(baseModel, 135)       
    compileModel(model, 1e-5)     
    
    # MODEL TUNING
    tunningHistory = modelTunning(model, trainingHistory, train_generator, val_generator)
    
    # TUNNING HISTORY DIAGRAMS
    drawTunningHistoryDiagrams(tunningHistory)        
  
if __name__ == "__main__":
    main()
