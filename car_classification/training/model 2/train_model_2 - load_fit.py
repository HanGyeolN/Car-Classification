from keras import models, layers
from keras import backend
from keras.applications.inception_v3 import InceptionV3
import matplotlib.pyplot as plt
import datetime
import os
import uuid
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
from scipy import misc
from keras.models import load_model

###############################
# 분류 CNN 학습 및 테스트
###############################


W = 300
H = 200

model = load_model('model')
model.load_weights('model_weight.h5')

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

train_datagen = ImageDataGenerator(rescale=1./255,
                                   width_shift_range=0.1,
                                   height_shift_range=0.1,
                                   horizontal_flip=True,shear_range=0.2)
valid_datagen = ImageDataGenerator(rescale=1./255)
train_path = 'dataset2/train'
valid_path = 'dataset2/valid'
train_batches = train_datagen.flow_from_directory(train_path, target_size=(H,W),
                                                  batch_size=10, class_mode='categorical')
valid_batches = valid_datagen.flow_from_directory(valid_path, target_size=(H,W),
                                                  batch_size=10, class_mode='categorical')

###########################################
max_acc = 0.88

for i in range(50):
    history = model.fit_generator(train_batches,
                                  steps_per_epoch=73,
                                  epochs=1,
                                  validation_data=valid_batches,
                                  validation_steps=5)

    list_acc = history.history['val_acc']
    acc = list_acc[0]

    if acc > max_acc:
        max_acc = acc
        model.save_weights('model2_weight.h5')
        model.save('model2')