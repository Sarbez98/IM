import sys
import os
import tensorflow as tf
from tensorflow import keras
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dropout, Flatten, Dense, Activation
from tensorflow.python.keras.layers import  Convolution2D, MaxPooling2D, Conv2D
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.callbacks import ModelCheckpoint

K.clear_session()

#tienes que instalar esta lib
#pip install pillow

print("---------------------------------")

# paths relativos mejor para portar codigo
path=os.getcwd()
data_entrenamiento = path+os.sep+'data/entrenamiento'
data_validacion =  path+os.sep+'data/validacion'#'.data/validación'
data_test =  path+os.sep+'data/test'#'.data/validación'

"""
Parameters
"""
longitud, altura = 150, 150
batch_size = 32
pasos = 100
filtrosConv1 = 32
filtrosConv2 = 64
tamano_filtro1 = (3, 3)
tamano_filtro2 = (2, 2)
tamano_pool = (2, 2)
clases = 3

##Preparamos nuestras imagenes
test_datagen = ImageDataGenerator(rescale=1. / 255)    
test_generador = test_datagen.flow_from_directory(
    data_test,
    target_size=(altura, longitud),
    batch_size=batch_size,
    class_mode='categorical')
    

#definimos el modelo
cnn = Sequential()
cnn.add(Conv2D(filtrosConv1, tamano_filtro1, padding ="same", input_shape=(longitud, altura, 3), activation='relu'))
cnn.add(MaxPooling2D(pool_size=tamano_pool))

cnn.add(Conv2D(filtrosConv2, tamano_filtro2, padding ="same",activation='relu'))
cnn.add(MaxPooling2D(pool_size=tamano_pool))

cnn.add(Conv2D(filtrosConv2, tamano_filtro2, padding ="same",activation='relu'))
cnn.add(MaxPooling2D(pool_size=tamano_pool))

cnn.add(Conv2D(filtrosConv2, tamano_filtro2, padding ="same",activation='relu'))
cnn.add(MaxPooling2D(pool_size=tamano_pool))

cnn.add(Flatten())
cnn.add(Dense(512, activation='relu'))
cnn.add(Dropout(0.5))                                        #desactiva neuronas y aprende otros patrones
cnn.add(Dense(256, activation='relu'))                      #otra capa de flatten y dense para que tenga capacidad de aprender cosas mas complejas, puede hacer sobreajute
cnn.add(Dropout(0.5))
cnn.add(Dense(128, activation='relu'))
cnn.add(Dropout(0.5))
cnn.add(Dense(clases, activation='softmax'))
cnn.summary()
cnn.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
 
# cargamos el modelo 
filepath="modelo.h5"				
cnn.load_weights(filepath)            

# evaluamos el modelo
score=cnn.evaluate(test_generador)

print('Test loss:', score[0])
print('Test accuracy:', score[1])

# Resultados

# Test loss: 0.08659811183072937
# Test accuracy: 0.976






