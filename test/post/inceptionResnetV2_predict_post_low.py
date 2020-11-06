"""
Implementation of InceptionResNetV2 in Keras

Reference:
C. Szegedy, V. Vanhoucke, S. Ioffe, J. Shlens and Z. Wojna, "Rethinking the Inception Architecture for Computer Vision," 2016 IEEE Conference on Computer Vision and Pattern Recognition (CVPR), Las Vegas, NV, 2016, pp. 2818-2826.
doi: 10.1109/CVPR.2016.308. 

Szegedy, Christian & Ioffe, Sergey & Vanhoucke, Vincent. (2016). Inception-v4, Inception-ResNet and the Impact of Residual Connections on Learning. AAAI Conference on Artificial Intelligence.
"""

from __future__ import print_function
# to filter some unnecessory warning messages
import warnings
warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")

from keras.layers import Input, Flatten, Dense, Dropout, GlobalAveragePooling2D
from keras.models import Model
from keras.optimizers import SGD, Adam, Adagrad 
from keras.regularizers import l1, l2
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.applications import InceptionResNetV2
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras.callbacks import ReduceLROnPlateau

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from keras.models import load_model

# USE ONLY ONE GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

XSIZE = 200
YSIZE = 300
ZSIZE = 3
train_dir="/home/hj/ss/B-mode-data/train/"
valid_dir="/home/hj/ss/B-mode-data/validation/"
#epochs=5000
#epochs=10
#epochs=50
#epochs=100
epochs=100
shuffle=False

def lr_schedule(epoch):
    """Learning Rate Schedule
    Learning rate is scheduled to be reduced after 80, 120, 160, 180 epochs.
    Called automatically every epoch as part of callbacks during training.
    # Arguments
        epoch (int): The number of epochs
    # Returns
        lr (float32): learning rate
    """
    """
    lr = 0.003
    if epoch > 500:
        lr *= 0.5e-3
    elif epoch > 300:
        lr *= 1e-3
    elif epoch > 200:
        lr *= 1e-2
    elif epoch > 100:
        lr *= 1e-1
    """
    lr = 0.001
    if epoch > 500:
        lr *= 0.5e-3
    elif epoch > 400:
        lr *= 1e-3
    elif epoch > 300:
        lr *= 1e-2
    elif epoch > 200:
        lr *= 1e-1
    print('Learning rate: ', lr)
    return lr

# ------------------------------------------------------------------ model ------------------------------------------------------------------
'''
model_ir2 = InceptionResNetV2(include_top=False, weights="imagenet", input_shape=(XSIZE, YSIZE, ZSIZE))
for layer in model_ir2.layers:
    layer.trainable = True
model = GlobalAveragePooling2D(name='GlobalAverage')(model_ir2.output)
model = Dense(256, activation='relu', kernel_regularizer=l2(0.04))(model)
model = Dropout(0.5)(model)
model = Dense(1, activation='sigmoid')(model)
model_ir2_sp = Model(model_ir2.input, model)
model_ir2_sp.compile(loss='binary_crossentropy', optimizer=Adam(lr=lr_schedule(0)), metrics=['acc'])
model_ir2_sp.summary()
'''
checkpoint_dir = "/home/hj/nose/waterT1C-20200923/post-0.001-2/InceptionResNetV2_model.091.h5"
model_ir2_sp = load_model(checkpoint_dir)
print("Created model and loaded weights from file")
# Compile model (required to make predictions)
#model_ir2_sp.compile(loss='binary_crossentropy', optimizer=Adam(lr=lr_schedule(0)), metrics=['acc'])

# ------------------------------------------------------------------ model:Checkpoint ------------------------------------------------------------------
'''
# Prepare model model saving directory.
save_dir = os.path.join(os.getcwd(), 'saved_models')
model_name = 'InceptionResNetV2_model.{epoch:03d}.h5'
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
filepath = os.path.join(save_dir, model_name)

# Prepare callbacks for model saving and for learning rate adjustment.
checkpoint = ModelCheckpoint(filepath=filepath,
                             monitor='val_acc',
                             verbose=1,
                             save_best_only=True)
lr_scheduler = LearningRateScheduler(lr_schedule)
lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1),
                               cooldown=0,
                               patience=10,
                               min_lr=0.5e-6)
callbacks = [checkpoint,lr_scheduler,lr_reducer]
'''
# ------------------------------------------------------------------ data ------------------------------------------------------------------
'''
# data augmentation
# This will do preprocessing and realtime data augmentation:
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    fill_mode='nearest',
    horizontal_flip=True,
    vertical_flip=True)
valid_datagen = ImageDataGenerator(rescale=1./255)

# Flow training images in batches 
train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(XSIZE, YSIZE),
        batch_size=32,
        class_mode='binary')

# Flow validation images in batches
valid_generator = valid_datagen.flow_from_directory(
        valid_dir,
        target_size=(XSIZE, YSIZE),
        batch_size=32,
        class_mode='binary')
'''






# pre
#data_dir_path = "/home/hj/nose/waterT1C-20200923/data/pre/test/lowrisk/"
#data_dir_path = "/home/hj/nose/waterT1C-20200923/data/pre/test/highrisk/"
# post
data_dir_path = "/home/hj/nose/waterT1C-20200923/data/post/test/lowrisk/"
#data_dir_path = "/home/hj/nose/waterT1C-20200923/data/post/test/highrisk/"


for f in os.listdir(data_dir_path):
    image_path = os.path.join(data_dir_path, f)
    img = image.load_img(image_path, target_size=(XSIZE, YSIZE))
    x = image.img_to_array(img) / 255.
    x = x.reshape((1,) + x.shape)
    print(f,",",model_ir2_sp.predict(x))
# ------------------------------------------------------------------ Fit model ------------------------------------------------------------------
'''
# Fit the model on the batches generated by datagen.flow().
history = model_ir2_sp.fit_generator(
      train_generator,
      steps_per_epoch=10, 
      epochs=epochs,
      validation_data=valid_generator,
      validation_steps=2,
      shuffle=shuffle,
      callbacks=callbacks,
      verbose=1)
'''

# ------------------------------------------------------------------ Plot:acc,loss ------------------------------------------------------------------
'''
# Plot training & validation accuracy values
#plt.figure()
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train_acc', 'Val_acc'], loc='upper left')
plt.savefig('model_training_acc.png')

# Plot training & validation loss values
#plt.figure()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train_acc', 'Val_acc', 'Train_loss', 'Val_loss'], loc='upper left')
plt.savefig('model_training_loss.png')
'''

