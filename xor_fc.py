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
error_per_epoch = net.train(x_train, y_train, epochs=35, learning_rate=0.1)


# test
out = net.predict_output(x_train)
print(out)


# visualize error per each epoch
epochs = 35
draw([k+1 for k in range(epochs)], error_per_epoch, 'Epochs', 'Error')

# save model
model = [x_train, y_train, out]
save_model('D:/Projects/PyloXyloto/saved models', 'xor_fc', model)
