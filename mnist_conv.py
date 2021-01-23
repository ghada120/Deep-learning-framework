# mnist example using convolution
from net import Network
from layers import *
from activations import tanh, tanh_derivative
from losses import mse, mse_derivative
from data import *
from metrics import *
from visualization import *
from utils import label_encoder

# load MNIST data
dataset, x, y = load_data('D:/Projects/DL Framework/data/train.csv')
x_train, x_test, y_train, y_test = split_data(dataset, 0.7)
# training data : 0.7*42000 samples

# normalize value to [0, 1]
x_train = normalize_data(x_train)
# x_train /= 255
# reshape the dataset into 4D array
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_train = x_train.astype('float64')
# encode output which is a number in range [0,9] into a vector of size 10
# e.g. number 3 will become [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
y_train = label_encoder(y_train)

# same for test data : 10000 samples
x_test = normalize_data(x_test)
# x_test /= 255
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
x_test = x_test.astype('float64')
y_test = label_encoder(y_test)

# Network
net = Network()
# input_shape=(28, 28, 1)   ;   output_shape=(28, 28, 1)
net.add_layer(ConvLayer(input_shape=(28, 28, 1), kernel_shape=(3, 3), layer_depth=1, stride=1, padding=1))
net.add_layer(ActivationLayer(tanh, tanh_derivative))
net.add_layer(FlattenLayer())                     # input_shape=(28, 28, 1)   ;   output_shape=(1, 28*28*1)
net.add_layer(FCLayer(28*28*1, 100))              # input_shape=(1, 28*28*1)  ;   output_shape=(1, 100)
net.add_layer(ActivationLayer(tanh, tanh_derivative))
net.add_layer(FCLayer(100, 10))                   # input_shape=(1, 100)      ;   output_shape=(1, 10)
net.add_layer(ActivationLayer(tanh, tanh_derivative))

# train on 1000 samples
# as we didn't implement mini-batch GD, training will be pretty slow if we update at each iteration on 60000 samples.
net.set_loss(mse, mse_derivative)
error_per_epoch = net.train(x_train[0:1000], y_train[0:1000], epochs=100, learning_rate=0.1)

# test on 10 samples
out = net.predict_output(x_test[0:10])
print("\n")
print("predicted values : ")
print(out, end="\n")
print("true values : ")
print(y_test[0:10])

# visualize error per each epoch
epochs = 100
visualize([k+1 for k in range(epochs)], error_per_epoch)

# conf_matrix = confusion_matrix(y_test[0:10], list(out)[0])
#
# print(conf_matrix)
# print(conf_matrix.values)
