import sys
import os
sys.path.append(os.getcwd())
from tensorflow import keras
import numpy as np
from modelConverter import Converter



base_model = keras.applications.resnet.ResNet50(include_top=False, weights=None, input_tensor=None, input_shape=(300,300,3), pooling=None, classes=1000)
y = keras.layers.Flatten()( base_model.layers[-1].output)
y = keras.layers.Dense(200, activation='relu')(y)
y = keras.layers.Dense(200, activation='relu')(y)
y = keras.layers.Dense(10, activation='softmax')(y)
x = base_model.layers[0].input
model = keras.models.Model(x,y)
model.compile(loss='categorical_crossentropy')

model.load_weights('model.h5')

#save model as pb 
Converter.pb.kerasmodel_to_pb(model, 'model/')

#convert .pb model into onnx format
Converter.onnx.pb_to_onnx('model/', 'model.onnx')

