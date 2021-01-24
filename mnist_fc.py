# xor example using FC Layer
from net import Network
from layers import FCLayer, ActivationLayer
from activations import tanh, tanh_derivative
from losses import mse, mse_derivative
from utils import *
from data import *
from visualization import *
from metrics import *
import pickle

# load MNIST data
dataset, x, y = load_data('D:/Projects/PyloXyloto/data/train.csv')
x_train, x_test, y_train, y_test = split_data(dataset, 0.7)
# training data : 0.7*42000 samples

# normalize value to [0, 1]
x_train = normalize_data(x_train)
# x_train /= 255
# reshape the dataset into 4D array
x_train = x_train.reshape(x_train.shape[0], 1, 28*28)
x_train = x_train.astype('float64')
# encode output which is a number in range [0,9] into a vector of size 10
# e.g. number 3 will become [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
y_train = label_encoder(y_train)

# same for test data : 10000 samples
x_test = normalize_data(x_test)
# x_test /= 255
x_test = x_test.reshape(x_test.shape[0], 1, 28*28)
x_test = x_test.astype('float64')
y_test = label_encoder(y_test)


# Network
net = Network()
net.add_layer(FCLayer(28*28, 100))                # input_shape=(1, 28*28)    ;   output_shape=(1, 100)
net.add_layer(ActivationLayer(tanh, tanh_derivative))
net.add_layer(FCLayer(100, 50))                   # input_shape=(1, 100)      ;   output_shape=(1, 50)
net.add_layer(ActivationLayer(tanh, tanh_derivative))
net.add_layer(FCLayer(50, 10))                    # input_shape=(1, 50)       ;   output_shape=(1, 10)
net.add_layer(ActivationLayer(tanh, tanh_derivative))

# train on 1000 samples
# as we didn't implemented mini-batch GD, training will be pretty slow if we update at each iteration on 60000 samples.
net.set_loss(mse, mse_derivative)
error_per_epoch = net.train(x_train[0:1000], y_train[0:1000], epochs=35, learning_rate=0.1)

# test on 10 samples and print predicted and true values
out = net.predict_output(x_test[0:10])
out_decoded_identity = identity_decoder(out)
out_decoded = label_decoder(out_decoded_identity)
label_decoded = label_decoder(y_test[0:10])
print("\n")
print("encoded predicted values : ")
print(out, end="\n")
print("identity encoded predicted values : ")
print(out_decoded_identity, end="\n")
print("predicted values : ")
print(out_decoded, end="\n")
print("\n")
print("identity encoded true values : ")
print(y_test[0:10], end="\n")
print("true values : ")
print(label_decoded, end="\n")

# calculate confusion matrix, accuracy, precision, recall, f1 score
conf_matrix = evaluation_metric(label_decoded, out_decoded, 'confusion matrix')
accuracy = evaluation_metric(label_decoded, out_decoded, 'accuracy')
precision = evaluation_metric(label_decoded, out_decoded, 'precision')
recall = evaluation_metric(label_decoded, out_decoded, 'recall')
f1 = evaluation_metric(label_decoded, out_decoded, 'f1')

print("confusion matrix:")
print(conf_matrix, end="\n")
# print(conf_matrix.values)
print("Total Accuracy: ", accuracy, end="\n")
print("Total Precision: ", precision, end="\n")
print("Total Recall: ", recall, end="\n")
print("F1 score: ", f1, end="\n")

# visualize error per each epoch
epochs = 35
draw([k+1 for k in range(epochs)], error_per_epoch, 'Epochs', 'Error')

# save model
model = [x_train, x_test, y_train, y_test, label_decoded, out_decoded]
save_model('D:/Projects/PyloXyloto/saved models', 'mnist_fc', model)

