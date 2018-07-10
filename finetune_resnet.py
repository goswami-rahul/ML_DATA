
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
np.random.seed(123)  # for reproducibility

from keras.models import Sequential, load_model, model_from_json, Model
from keras.applications.resnet50 import preprocess_input, ResNet50
from keras.layers import Dense, Dropout, Activation, Flatten, BatchNormalization
from keras.layers import MaxPooling2D, Conv2D
from keras.utils import np_utils
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array, array_to_img
from keras.callbacks import ModelCheckpoint, TensorBoard, ReduceLROnPlateau, EarlyStopping
from keras.optimizers import SGD
from keras.metrics import categorical_accuracy

import tensorflow as tf
# run_opts = tf.RunOptions(report_tensor_allocations_upon_oom = True)


# In[2]:


CLASS_MAP = {'antelope': 0,
 'bat': 1,
 'beaver': 2,
 'bobcat': 3,
 'buffalo': 4,
 'chihuahua': 5,
 'chimpanzee': 6,
 'collie': 7,
 'dalmatian': 8,
 'german+shepherd': 9,
 'grizzly+bear': 10,
 'hippopotamus': 11,
 'horse': 12,
 'killer+whale': 13,
 'mole': 14,
 'moose': 15,
 'mouse': 16,
 'otter': 17,
 'ox': 18,
 'persian+cat': 19,
 'raccoon': 20,
 'rat': 21,
 'rhinoceros': 22,
 'seal': 23,
 'siamese+cat': 24,
 'spider+monkey': 25,
 'squirrel': 26,
 'walrus': 27,
 'weasel': 28,
 'wolf': 29}
CLASS_WEIGHTS = [0.6235012 , 1.69270833, 3.25814536, 1.03668262, 0.71507151,
       1.12262522, 0.90845563, 0.6372549 , 1.20705664, 0.63076177,
       0.74328188, 0.93390805, 0.390039  , 2.24525043, 7.22222222,
       0.91036415, 3.49462366, 0.83493899, 0.86493679, 0.88255261,
       1.25240848, 1.96969697, 0.90845563, 0.65162907, 1.27077224,
       2.29276896, 0.53630363, 2.92792793, 2.35507246, 1.07526882]


# In[3]:


batch_size = 16
resol = 224

# this is the augmentation configuration we will use for training
train_datagen = ImageDataGenerator(
        rotation_range=15,
        width_shift_range=0.05,
        height_shift_range=0.05,
        shear_range=0.5,
        zoom_range=0.1,
        horizontal_flip=True,
        preprocessing_function=preprocess_input,
        )
# this is the augmentation configuration we will use for testing:
# only rescaling
test_datagen = ImageDataGenerator(horizontal_flip=True,
                                  preprocessing_function=preprocess_input)

train_generator = train_datagen.flow_from_directory(
        'train_data/',  # this is the target directory
        target_size=(resol, resol),
        batch_size=batch_size,
        class_mode='categorical')


# # this is a similar generator, for validation data
validation_generator = test_datagen.flow_from_directory(
        'test_data/',
        target_size=(resol, resol),
        batch_size=batch_size,
        class_mode='categorical')


# In[4]:


clsdict =  {v: k for k, v in train_generator.class_indices.items()}
cnames = os.listdir("train_data")
cnames.sort()


# In[5]:


def load_top_model():
    model = Sequential()
    model.add(Dense(1024, activation='relu', input_shape=(2048,)))
    model.add(BatchNormalization())
    model.add(Dropout(0.1))
    model.add(Dense(128, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.1))
    model.add(Dense(30, activation='sigmoid'))
#     opt = 'adam'
#     opt = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
#     model.compile(optimizer=opt,
#                   loss='categorical_crossentropy',
#                   metrics=[categorical_accuracy])
#     model.load_weights("MODEL/top_model_initial.h5")
    return model


# In[6]:


input_model = ResNet50(include_top=False, pooling='avg')
top = load_top_model()


# In[7]:


x = top(input_model.outputs)
model = Model(inputs=input_model.inputs, outputs=x)


# In[8]:


len(input_model.layers)


# In[9]:


for layer in input_model.layers[:-15]:
    layer.trainable = False
model.summary()


# In[10]:


top.summary()


# In[13]:


my_callbacks = [TensorBoard(batch_size=batch_size)]
my_callbacks.append(ModelCheckpoint(filepath="checkpoints/weights.{epoch:02d}-{val_loss:.2f}.hdf5", verbose=1))

reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.3,
                              patience=3, min_lr=0.0000003)
early_stops = EarlyStopping(monitor='val_loss',
                patience=15,
                verbose=1,)
my_callbacks.append(reduce_lr)
my_callbacks.append(early_stops)
my_callbacks


# In[14]:


opt = SGD(lr=0.0002, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(optimizer=opt,
              loss='categorical_crossentropy',
              metrics=[categorical_accuracy])
model.load_weights("MODEL/ResNet50/weights_0.7965.h5")


# In[15]:


history = model.fit_generator(
        train_generator,
        steps_per_epoch=10412 // batch_size,
        epochs=500,
        verbose=1,
        callbacks=my_callbacks,
        validation_data=validation_generator,
        validation_steps=2588 // batch_size,
        class_weight=CLASS_WEIGHTS,
        )
model.save('MODEL/ResNet50/transfer_learn_finetune.h5')
pd.DataFrame(history.history).to_csv("RESULTS/ResNet50/history_transfer_learn_finetune.csv")


# In[17]:


def plot_history(history):
    loss_list = [s for s in history.keys() if 'acc' not in s and 'val' not in s]
    val_loss_list = [s for s in history.keys() if 'acc' not in s and 'val' in s]
    acc_list = [s for s in history.keys() if 'acc' in s and 'val' not in s]
    val_acc_list = [s for s in history.keys() if 'acc' in s and 'val' in s]

    if len(loss_list) == 0:
        print('Loss is missing in history')
        return

    ## As loss always exists
    epochs = range(1,len(history[loss_list[0]]) + 1)

    ## Loss
    for ll, vl in zip(loss_list, val_loss_list):
        plt.figure()
        plt.plot(epochs, history[ll], 'b', label=f'{ll} ({history[ll][-1]:.5f})')
        plt.plot(epochs, history[vl], 'g', label=f'{vl} ({history[vl][-1]:.5f})')

        plt.title('Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()

    ## Accuracy
    for ll, vl in zip(acc_list, val_acc_list):
        plt.figure()
        plt.plot(epochs, history[ll], 'b', label=f'{ll} ({history[ll][-1]:.5f})')
        plt.plot(epochs, history[vl], 'g', label=f'{vl} ({history[vl][-1]:.5f})')
        plt.title('Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()
    plt.show()


# In[18]:


plot_history(history.history)
