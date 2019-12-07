from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
import pickle as pkl

import sys
import numpy as np
np.random.seed(0)

# from load_data import load
from main2 import CustomMultiLossLayer

train = pkl.load(open('train.pkl', 'rb'))
test = pkl.load(open('test.pkl', 'rb'))

train, test = train / 255.0, test / 255.0

X_train = train['images']
y1_train = train['instance']
y2_train = train['label']

X_test = test['images']
y1_test = test['instance']
y2_test = test['label']

N = 100
epochs = 10
batch_size = 20
nb_features = 1
Q = (32, 32, 4)
D1 = Q  # first output
D2 = Q  # second output

def get_prediction_model():
	inp = layers.Input(shape=Q, name='inp')
	x = layers.Conv2D(32, (3, 3), activation='relu', input_shape=Q, data_format='channels_first')(inp)
	x = layers.MaxPooling2D((2, 2))(x)
	x = layers.Conv2D(64, (3, 3), activation='relu')(x)
	x = layers.MaxPooling2D((2, 2))(x)
	x = layers.Conv2D(64, (3, 3), activation='relu')(x)
	x = layers.Flatten()(x)
	x = layers.Dense(64, activation='relu')(x)
	y1_pred = layers.Dense(64)(x)
	y1_pred = layers.Conv2DTranspose(input_shape=Q, filters=64, kernel_size=(3,3))(y1_pred)
	y2_pred = layers.Dense(64)(x)
	y2_pred = layers.Conv2DTranspose(input_shape=Q, filters=64, kernel_size=(3,3))(y2_pred)
	return models.Model(inp, [y1_pred, y2_pred])

def get_trainable_model(prediction_model):
	inp = layers.Input(shape=Q, name='inp')
	y1_pred, y2_pred = prediction_model(inp)
	y1_true = layers.Input(shape=D1, name='y1_true')
	y2_true = layers.Input(shape=D2, name='y2_true')
	out = CustomMultiLossLayer(nb_outputs=2)([y1_true, y2_true, y1_pred, y2_pred])
	return models.Model([inp, y1_true, y2_true], out)

# model = models.Sequential()
# model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(16, 32, 4)))
# model.add(layers.MaxPooling2D((2, 2)))
# model.add(layers.Conv2D(64, (3, 3), activation='relu'))
# model.add(layers.MaxPooling2D((2, 2)))
# model.add(layers.Conv2D(64, (3, 3), activation='relu'))
# model.add(layers.Flatten())
# model.add(layers.Dense(64, activation='relu'))
# # model.add(layers.Dense(10, activation='softmax'))
# model.add(CustomMultiLossLayer())

model = get_trainable_model(get_prediction_model())

# model.summary()

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# history = model.fit(X_train, y_train, epochs=10, 
#                     validation_data=(test_images, test_labels))
history = trainable_model.fit([X_train, y1_train, y2_train], epochs=epochs, batch_size=batch_size, verbose=0)

pylab.plot(hist.history['loss'])
print([np.exp(K.get_value(log_var[0]))**0.5 for log_var in trainable_model.layers[-1].log_vars])

model.save('multitask.h5')

plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.5, 1])
plt.legend(loc='lower right')

# test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)
# print


