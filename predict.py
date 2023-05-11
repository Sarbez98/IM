import os
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
from PIL import Image

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

#aplicamos a una imagen
img = Image.open(path+"/data/test/T1/T1_497_3.png")
img = img.resize((longitud,altura),Image.ANTIALIAS)
img = np.asarray(img)
img=img/255
print(img.shape)
x=np.zeros((longitud,altura,3))
x[:,:,0]=img
x[:,:,1]=img
x[:,:,2]=img
x = np.reshape(x,(1,longitud,altura,3))

array = cnn.predict(x)

result = array[0]
answer = np.argmax(result)
if answer == 0:
    print("pred: T1")
elif answer == 1:
    print("pred: T2")
elif answer == 2:
    print("pred: flair")

