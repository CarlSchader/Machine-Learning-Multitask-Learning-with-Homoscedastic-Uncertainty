import sys
import numpy as np
np.random.seed(0)

from keras.layers import Input, Dense, Lambda, Layer
from keras.initializers import Constant
from keras.models import Model
from keras import backend as K
import pickle as pkl

# from load_data import load

# df = load("./gtFine/train/aachen/")

# Custom loss layer
class CustomMultiLossLayer(Layer):
    def __init__(self, nb_outputs=2, **kwargs):
        self.nb_outputs = nb_outputs
        self.is_placeholder = True
        super(CustomMultiLossLayer, self).__init__(**kwargs)
        
    def build(self, input_shape=None):
        # initialise log_vars
        self.log_vars = []
        for i in range(self.nb_outputs):
            self.log_vars += [self.add_weight(name='log_var' + str(i), shape=(1,),
                                              initializer=Constant(0.), trainable=True)]
        super(CustomMultiLossLayer, self).build(input_shape)

    def multi_loss(self, ys_true, ys_pred):
        assert len(ys_true) == self.nb_outputs and len(ys_pred) == self.nb_outputs
        loss = 0
        for y_true, y_pred, log_var in zip(ys_true, ys_pred, self.log_vars):
            precision = K.exp(-log_var[0])
            loss += K.sum(precision * (y_true - y_pred)**2. + log_var[0], -1)
        return K.mean(loss)

    def call(self, inputs):
        ys_true = inputs[:self.nb_outputs]
        ys_pred = inputs[self.nb_outputs:]
        loss = self.multi_loss(ys_true, ys_pred)
        self.add_loss(loss, inputs=inputs)
        # We won't actually use the output.
        return K.concatenate(inputs, -1)

N = 100
epochs = 2000
batch_size = 20
nb_features = 1024
Q = 1
D1 = 1  # first output
D2 = 1  # second output

def gen_data_linear(N, w1=2., b1=8., sigma1=1e1, w2=3, b2=3., sigma2=1e0):
    X = np.random.randn(N, Q)
    Y1 = X.dot(w1) + b1 + sigma1 * np.random.randn(N, D1)
    Y2 = X.dot(w2) + b2 + sigma2 * np.random.randn(N, D2)
    return X, Y1, Y2

def gen_data_quadratic(N, w1=2., b1=8., sigma1=1e1, w2=3, b2=3., sigma2=1e0):
	X = np.random.randn(N, Q)
    Y1 = X * X * w1 + b1 + sigma1 * np.random.randn(N, D1)
    Y2 = X * X * w2 + b2 + sigma2 * np.random.randn(N, D2)
    return X, Y1, Y2

def gen_data_both(N, w1=2., b1=8., sigma1=1e1, w2=3, b2=3., sigma2=1e0):
	X = np.random.randn(N, Q)
    Y1 = X.dot(w1) + b1 + sigma1 * np.random.randn(N, D1)
    Y2 = X * X * w2 + b2 + sigma2 * np.random.randn(N, D2)
    return X, Y1, Y2

import pylab
# %matplotlib inline

#  Change these data generation functions based on what you need.
if len(sys.argv) == 7:
	X, Y1, Y2 = gen_data(N, w1=float(sys.argv[1]), b1=float(sys.argv[2]), sigma1=float(sys.argv[3]), w2=float(sys.argv[4]), b2=float(sys.argv[5]), sigma2=float(sys.argv[6]))
else:
	X, Y1, Y2 = gen_data(N)
pylab.figure(figsize=(3, 1.5))
pylab.scatter(X[:, 0], Y1[:, 0])
pylab.scatter(X[:, 0], Y2[:, 0])
pylab.show()

def get_prediction_model():
    inp = Input(shape=(Q,), name='inp')
    x = Dense(nb_features, activation='relu')(inp)
    y1_pred = Dense(D1)(x)
    y2_pred = Dense(D2)(x)
    return Model(inp, [y1_pred, y2_pred])

def get_trainable_model(prediction_model):
    inp = Input(shape=(Q,), name='inp')
    y1_pred, y2_pred = prediction_model(inp)
    y1_true = Input(shape=(D1,), name='y1_true')
    y2_true = Input(shape=(D2,), name='y2_true')
    out = CustomMultiLossLayer(nb_outputs=2)([y1_true, y2_true, y1_pred, y2_pred])
    return Model([inp, y1_true, y2_true], out)

prediction_model = get_prediction_model()
trainable_model = get_trainable_model(prediction_model)
trainable_model.compile(optimizer='adam', loss=None)
assert len(trainable_model.layers[-1].trainable_weights) == 2  # two log_vars, one for each output
assert len(trainable_model.losses) == 1

print("train")
hist = trainable_model.fit([X, Y1, Y2], epochs=epochs, batch_size=batch_size, verbose=0)
print("trained")
pylab.plot(hist.history['loss'])

# Found standard deviations (ground truth is 10 and 1):
print([np.exp(K.get_value(log_var[0]))**0.5 for log_var in trainable_model.layers[-1].log_vars])
pkl.dump(trainable_model, open('simple_multitask.pkl', 'wb'))