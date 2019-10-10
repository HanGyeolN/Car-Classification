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

def save_history_history(fname, history_history, fold=''):
    np.save(os.path.join(fold, fname), history_history)

def unique_filename(type='uuid'):
    if type == 'datetime':
        filename = datetime.datetime.now().strftime("%y%m%d_%H%M%S")
    else:  # type == "uuid"
        filename = str(uuid.uuid4())
    return filename


###########################
# 학습 효과 분석02
###########################


def plot_val_acc(history, title=None):
    plt.plot(history.history['val_acc'])
    plt.title(title)
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Validation'], loc=0)

def plot_acc(history, title=None):
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title(title)
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Training', 'Validation'], loc=0)

def plot_loss(history, title=None):
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title(title)
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Training', 'Validation'], loc=0)

###############################
# 분류 CNN 학습 및 테스트
###############################

# size 915 360
def main():
    W = 300
    H = 200
    input_shape = [H, W, 3]
    number_of_class = 2
    Nout = number_of_class

    base_model = InceptionV3(weights='imagenet', include_top=False)
    x = base_model.output
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(200, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(200, activation='relu')(x)
    predictions = layers.Dense(Nout, activation='softmax')(x)
    model = models.Model(inputs=base_model.input, outputs=predictions)

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    train_datagen = ImageDataGenerator(rescale=1./255,
                                       width_shift_range=0.1,
                                       height_shift_range=0.1,
                                       horizontal_flip=True,shear_range=0.2)
    valid_datagen = ImageDataGenerator(rescale=1./255)
    test_datagen = ImageDataGenerator(rescale=1./255)

    train_path = 'dataset2/train'
    valid_path = 'dataset2/valid'
    test_path = 'dataset2/valid'

    train_batches = train_datagen.flow_from_directory(train_path, target_size=(H,W),
                                                      batch_size=10, class_mode='categorical')
    valid_batches = valid_datagen.flow_from_directory(valid_path, target_size=(H,W),
                                                      batch_size=10, class_mode='categorical')
    test_batches = test_datagen.flow_from_directory(test_path, target_size=(H,W),
                                                    batch_size=10, class_mode='categorical',shuffle=False)

    ###########################################
    history = model.fit_generator(train_batches,
                                  steps_per_epoch=73,
                                  epochs=75,
                                  validation_data=valid_batches,
                                  validation_steps=5)
    # 7. Test Error
    print("-- InceptionV3 validation ACCURACY --")
    scores = model.evaluate_generator(test_batches, steps=5)
    print("%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))

    # 8. 그래프 저장
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plot_acc(history)
    plt.subplot(1, 2, 2)
    plot_loss(history)
    savename = 'Problem2_InceptionV3.png'
    plt.savefig(savename)

    plt.figure(figsize=(6, 4))
    plot_val_acc(history)
    savename2 = 'Problem2_InceptionV3_val.png'
    plt.savefig(savename2)

    # 9. 모델 저장
    suffix = unique_filename('datetime')
    foldname = 'output_' + suffix
    os.makedirs(foldname)
    save_history_history('history_history.npy', history.history, fold=foldname)
    model.save_weights(os.path.join(foldname, 'model_weight.h5'))
    model.save(os.path.join(foldname, 'model'))

if __name__ == '__main__':
    main()