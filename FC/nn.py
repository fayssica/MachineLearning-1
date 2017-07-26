from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from six.moves import xrange


TRAIN_DATA = 'two_spiral_train.txt'
TEST_DATA = 'two_spiral_test.txt'

class Draw:
    def __init__(self):
        self.x = np.linspace(-5,5,100)
        self.y = np.linspace(-5,5,100)
        self.X, self.Y = np.meshgrid(self.x,self.y)
        self.X_f = self.X.flatten()
        self.Y_f = self.Y.flatten()
        self.p = zip(self.X_f, self.Y_f)
        self.data = list()
        for i in self.p:
            self.data.append(list(i))
        self.data = np.array(self.data)


    def draw3D(self, Z, angle):
        fig = plt.figure(figsize=(15,7))
        ax = Axes3D(fig)
        ax.view_init(angle[0], angle[1])
        ax.plot_surface(self.X,self.Y,Z,rstride=1, cstride=1, cmap='rainbow')
        plt.show()

    def draw2D(self, Z):
        plt.figure()
        plt.scatter(self.X_f,self.Y_f,c=Z)
        plt.show()


class FC:
    """Define a fully connected layer"""

    def __init__(self, input_dim, output_dim, lr):
        self._input_dim = input_dim
        self._output_dim = output_dim
        self.lr = lr
        self.w = np.random.randn(input_dim, output_dim)
        self.b = np.zeros(output_dim)

    def _sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def forward(self, x):
        self.y = self._sigmoid(np.dot(x, self.w) + self.b)
        self.x = x
        return self.y

    def backward(self, gradient):
        grad_z = gradient * self.y * (1 - self.y)
        grad_w = np.dot(self.x.T, grad_z)
        grad_b = grad_z.sum(0)
        grad_x = np.dot(grad_z, self.w.T)
        self.w -= grad_w * self.lr
        self.b -= grad_b * self.lr
        return grad_x


class SquareLoss:
    """Define the loss function"""

    def forward(self, output, label):
        self.loss = output - label
        return np.sum(self.loss * self.loss) / self.loss.shape[0] / 2

    def backward(self):
        return self.loss

    def accuracy(self, output, label):
        return (np.around(output) == label).sum() / len(label)


class Net:
    """Train the model"""

    def __init__(self, input_dim, hidden_dim, output_dim, lr):
        self.fc1 = FC(input_dim, hidden_dim, lr)
        self.fc2 = FC(hidden_dim, output_dim, lr)
        self.loss = SquareLoss()

    def train(self, train_data, train_label, iter):
        for i in xrange(iter):
            # forward step
            out_fc1 = self.fc1.forward(train_data)
            out_fc2 = self.fc2.forward(out_fc1)
            out_loss = self.loss.forward(out_fc2, train_label)
            # backward step
            loss_grad = self.loss.backward()
            loss_fc2 = self.fc2.backward(loss_grad)
            loss_fc1 = self.fc1.backward(loss_fc2)
            if i % 10 == 0:
                train_accuracy = self.loss.accuracy(out_fc2, train_label)
                print("Iter: {0}   Train accuracy: {1}".format(
                    i, train_accuracy))

    def test(self, test_data, test_label):
        out_fc1 = self.fc1.forward(test_data)
        out_fc2 = self.fc2.forward(out_fc1)
        out_loss = self.loss.forward(out_fc2, test_label)
        accuracy = self.loss.accuracy(out_fc2, test_label)
        return accuracy

    def predict(self, predict_data):
        out_fc1 = self.fc1.forward(predict_data)
        out_fc2 = self.fc2.forward(out_fc1)
        out_result = np.around(out_fc2)
        return out_result


def main():
    train_set=np.loadtxt(TRAIN_DATA)
    test_set=np.loadtxt(TEST_DATA)
    train_data = train_set[:, :2]
    train_label = train_set[:, 2].reshape((-1, 1))
    test_data = test_set[:, :2]
    test_label = test_set[:, 2].reshape((-1, 1))

    net = Net(2, 10, 1, 0.1)
    net.train(train_data, train_label, 5000)
    accuracy = net.test(test_data, test_label)
    print('Test accuracy: {0}'.format(accuracy))

    draw = Draw()
    out = net.predict(draw.data)
    draw.draw2D(out)
    #out = out.reshape((100,100))
    #draw.draw3D(out,(40,-45))

if __name__ == '__main__':
    main()