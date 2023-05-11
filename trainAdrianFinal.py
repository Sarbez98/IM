import sys
import os
import tensorflow as tf
from tensorflow import keras
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dropout, Flatten, Dense, Activation
from tensorflow.python.keras.layers import  Convolution2D, MaxPooling2D, Conv2D
from tensorflow.python.keras import backend as K
from keras.optimizers import Adam
from tensorflow.python.keras.callbacks import ModelCheckpoint, EarlyStopping
import numpy as np

K.clear_session()

#tienes que instalar esta lib
#pip install pillow

print("---------------------------------")

# paths relativos mejor para portar codigo
path=os.getcwd()
data_entrenamiento = path+os.sep+'data/entrenamiento'
data_validacion =  path+os.sep+'data/validacion'
data_test =  path+os.sep+'data/test'

"""
Parameters
"""
epocas=100
longitud, altura = 150, 150
batch_size = 32
pasos = 90
validation_steps = 30
filtrosConv1 = 32
filtrosConv2 = 64
tamano_filtro1 = (3, 3)
tamano_filtro2 = (2, 2)
tamano_pool = (2, 2)
clases = 3
learning_rate = 0.001# no se usa, estamos usando adam que tiene 0.001 por defecto
#si queremos mod tenemos que hacer:
#opt = keras.optimizers.Adam(lr=learning_rate)
#cnn.compile(loss='categorical_crossentropy', 
 #           optimizer=opt, 
  #          metrics=['accuracy'])

##Preparamos nuestras imagenes

entrenamiento_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    preprocessing_function=lambda x: x + np.random.normal(0.0, 0.1, x.shape))#añadir ruido ya, evita sobreajuste)

val_datagen = ImageDataGenerator(rescale=1. / 255)
test_datagen = ImageDataGenerator(rescale=1. / 255)

entrenamiento_generador = entrenamiento_datagen.flow_from_directory(
    data_entrenamiento,
    target_size=(altura, longitud),
    batch_size=batch_size,
    class_mode='categorical')
    
validacion_generador = val_datagen.flow_from_directory(
    data_validacion,
    target_size=(altura, longitud),
    batch_size=batch_size,
    class_mode='categorical')
   
    
test_generador = test_datagen.flow_from_directory(
    data_test,
    target_size=(altura, longitud),
    batch_size=batch_size,
    class_mode='categorical')
    

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

# path absoluto por si el generador cambia de diractorio
path=os.getcwd()
filepath=path+os.sep+'modelo.h5'		

try:
  cnn.load_weights(filepath)
except:
    print("load errror")	


savemodel =ModelCheckpoint(filepath, monitor='val_loss', save_best_only=True, save_weights_only=True, mode='min', period=1)
early=EarlyStopping(monitor='val_loss', patience=10)                                                                               

cnn.compile(loss='categorical_crossentropy', 
            optimizer='adam', 
            metrics=['accuracy'])


cnn.fit(#antes fit_generator
    entrenamiento_generador,
    steps_per_epoch=pasos,
    epochs=epocas,
    validation_data=validacion_generador,
    validation_steps=validation_steps,
    callbacks=[savemodel,early])

#target_dir = 'modelo'
#if not os.path.exists(target_dir):
#  os.mkdir(target_dir)
#cnn.save('modelo.h5')
#cnn.save_weights('pesos.h5')

cnn.load_weights(filepath)	

score=cnn.evaluate(test_generador)

print('Test loss:', score[0])
print('Test accuracy:', score[1])


# try:
    # # Ejecutar normalmente hasta que se interrumpa manualmente
    # while True:
        # pass
# except KeyboardInterrupt:
    # # Capturar la excepción y guardar el modelo antes de salir
    # cnn.save(filepath)
    # print("\nModelo guardado en", os.path.abspath(filepath))

# Test loss: 
# Test accuracy: 


#subo patiene 15 
