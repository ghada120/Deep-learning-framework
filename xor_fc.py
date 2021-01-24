# xor example
from net import Network
from layers import *
from activations import *
from losses import *
from metrics import *
from visualization import *
from utils import *

# training data
x_train = np.array([[[0,0]], [[0,1]], [[1,0]], [[1,1]]])
y_train = np.array([[[0]], [[1]], [[1]], [[0]]])

# network
net = Network()
net.add_layer(FCLayer(2, 3))
net.add_layer(ActivationLayer(tanh, tanh_derivative))
net.add_layer(FCLayer(3, 1))
net.add_layer(ActivationLayer(tanh, tanh_derivative))

# train
net.set_loss(mse, mse_derivative)
error_per_epoch, accuracy_per_epoch = net.train(x_train, y_train, epochs=1000, learning_rate=0.1)


# test
out = net.predict_output(x_train)
out_decoded = label_decoder(out)
label_decoded = list(np.squeeze(y_train))
print(out, end="\n")
print(out_decoded, end="\n")
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

# save model
model = [x_train, y_train, out]
save_model('D:/Projects/PyloXyloto/models', 'xor_fc', model)
