import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt

####################################

class ReLULayer(object):
    def forward(self, input):
        # remember the input for later backpropagation
        self.input = input
        # return the ReLU of the input
        relu = np.where(self.input > 0, self.input, 0) # your code here
        return relu

    def backward(self, upstream_gradient):
        # compute the derivative of ReLU from upstream_gradient and the stored input
        downstream_gradient = upstream_gradient * np.where(self.input > 0, 1, 0) # your code here
        return downstream_gradient

    def update(self, learning_rate):
        pass # ReLU is parameter-free

####################################

class OutputLayer(object):
    def __init__(self, n_classes):
        self.n_classes = n_classes

    def forward(self, input):
        # remember the input for later backpropagation
        self.input = input
        # return the softmax of the input
        softmax = np.exp(self.input)/np.sum(np.exp(self.input), axis=1, keepdims=True) # your code here
        return softmax

    def backward(self, predicted_posteriors, true_labels):
        # return the loss derivative with respect to the stored inputs
        # (use cross-entropy loss and the chain rule for softmax,
        #  as derived in the lecture)
        downstream_gradient = np.where(true_labels == 1, predicted_posteriors-1, predicted_posteriors)/predicted_posteriors.shape[0] # Why divide here by predicted_posteriors.shape[0]
        return downstream_gradient

    def update(self, learning_rate):
        pass # softmax is parameter-free

####################################

class LinearLayer(object):
    def __init__(self, n_inputs, n_outputs):
        self.n_inputs  = n_inputs
        self.n_outputs = n_outputs
        # randomly initialize weights and intercepts
        # Following the lecture's proposal for random initialization and ReLU activation
        self.B = np.random.normal(0, np.sqrt(2/(n_inputs+1)), size = (n_inputs, n_outputs)) # your code here
        self.b = np.random.normal(0, np.sqrt(2/(n_inputs+1)), size = n_outputs) # your code here

    def forward(self, input):
        # remember the input for later backpropagation
        self.input = input
        # compute the scalar product of input and weights
        # (these are the preactivations for the subsequent non-linear layer)
        preactivations = self.input @ self.B + self.b # your code here
        return preactivations

    def backward(self, upstream_gradient):
        # compute the derivative of the weights from
        # upstream_gradient and the stored input
        self.grad_b = np.sum(upstream_gradient, axis = 0) # Sum because of summing via all batches
        self.grad_B = self.input.T @ upstream_gradient # @ carries out sum matrix product -> Includes summation of all batches
        # compute the downstream gradient to be passed to the preceding layer
        downstream_gradient = upstream_gradient @ self.B.T # your code here -> Might be self.B.T instead -> Because derivative drops out (see Lecture)
        return downstream_gradient

    def update(self, learning_rate):
        # update the weights by batch gradient descent
        self.B = self.B - learning_rate * self.grad_B
        self.b = self.b - learning_rate * self.grad_b

####################################

class MLP(object):
    def __init__(self, n_features, layer_sizes):
        # constuct a multi-layer perceptron
        # with ReLU activation in the hidden layers and softmax output
        # (i.e. it predicts the posterior probability of a classification problem)
        #
        # n_features: number of inputs
        # len(layer_size): number of layers
        # layer_size[k]: number of neurons in layer k
        # (specifically: layer_sizes[-1] is the number of classes)
        self.n_layers = len(layer_sizes)
        self.layers   = []

        # create interior layers (linear + ReLU)
        n_in = n_features
        for n_out in layer_sizes[:-1]:
            self.layers.append(LinearLayer(n_in, n_out))
            self.layers.append(ReLULayer())
            n_in = n_out

        # create last linear layer + output layer
        n_out = layer_sizes[-1]
        self.layers.append(LinearLayer(n_in, n_out))
        self.layers.append(OutputLayer(n_out))

    def forward(self, X):
        # X is a mini-batch of instances
        batch_size = X.shape[0]
        # flatten the other dimensions of X (in case instances are images)
        X = X.reshape(batch_size, -1)

        # compute the forward pass
        # (implicitly stores internal activations for later backpropagation)
        result = X
        for layer in self.layers:
            result = layer.forward(result)
        return result

    def backward(self, predicted_posteriors, true_classes):
        
        # we use one-hot encoding (so the shape of true_labels is compatible with the posteriors in the layer backpropagation)
        true_labels = np.zeros_like(predicted_posteriors)
        true_labels[np.arange(len(true_classes)), true_classes] = 1

        # perform backpropagation w.r.t. the prediction for the latest mini-batch X
        upstream_gradient = self.layers[-1].backward(predicted_posteriors, true_labels)
        for layer in reversed(self.layers[:-1]):
            # we iterate backwards
            upstream_gradient = layer.backward(upstream_gradient) # your code here

    def update(self, X, Y, learning_rate):
        posteriors = self.forward(X)
        self.backward(posteriors, Y)
        for layer in self.layers:
            layer.update(learning_rate)

    def train(self, x, y, n_epochs, batch_size, learning_rate):
        N = len(x)
        n_batches = N // batch_size
        for i in range(n_epochs):
            # print("Epoch", i)
            # reorder data for every epoch
            # (i.e. sample mini-batches without replacement)
            permutation = np.random.permutation(N)

            for batch in range(n_batches):
                # create mini-batch
                start = batch * batch_size
                x_batch = x[permutation[start:start+batch_size]]
                y_batch = y[permutation[start:start+batch_size]]

                # perform one forward and backward pass and update network parameters
                self.update(x_batch, y_batch, learning_rate)

##################################

if __name__=="__main__":

    # set training/test set size
    N = 2000

    # create training and test data
    X_train, Y_train = datasets.make_moons(N, noise=0.05)
    X_test,  Y_test  = datasets.make_moons(N, noise=0.05)
    n_features = 2
    n_classes  = 2

    # standardize features to be in [-1, 1]
    offset  = X_train.min(axis=0)
    scaling = X_train.max(axis=0) - offset
    X_train = ((X_train - offset) / scaling - 0.5) * 2.0
    X_test  = ((X_test  - offset) / scaling - 0.5) * 2.0

    # set hyperparameters (play with these!)
    layer_sizes = [4, 4, n_classes]
    n_epochs = 100
    batch_size = 20
    learning_rate = 0.05

    # create network
    network = MLP(n_features, layer_sizes)

    # train
    network.train(X_train, Y_train, n_epochs, batch_size, learning_rate)

    # test
    predicted_posteriors = network.forward(X_test)
    # determine class predictions from posteriors by winner-takes-all rule
    predicted_classes = np.argmax(predicted_posteriors, axis=1) # your code here
    # compute and output the error rate of predicted_classes
    error_rate = np.mean(predicted_classes != Y_test) # your code here
    print("error rate:", error_rate)

for run in [2,5,10,30]:
    # compute for different layer sizes (as indicated in the task)
    layer_sizes = [run,run,n_classes]
    network = MLP(n_features,layer_sizes)
    #train
    network.train(X_train, Y_train, n_epochs, batch_size, learning_rate)
    # test
    predicted_posteriors = network.forward(X_test)
    predicted_posteriors_train = network.forward(X_train)
    # determine class predictions from posteriors by winner-takes-all rule
    predicted_classes = np.argmax(predicted_posteriors, axis=1)
    predicted_train = np.argmax(predicted_posteriors_train, axis=1)
    # compute and output the error rate of predicted_classes
    error_rate = np.mean(predicted_classes != Y_test)
    print("error rate for {:.0f}-layers:".format(run), error_rate)
    

#Visualoze decision boundary and half-moon instances
xx,yy = np.meshgrid(np.linspace(-1.1,1.1,1000),np.linspace(-1.1,1.1,1000))
Z = np.argmax(network.forward(np.c_[xx.ravel(),yy.ravel()]),axis=1)
Z = Z.reshape(xx.shape)
plt.figure(figsize=(np.sqrt(2)*5,5))
plt.title("Decision boundary and instances")
plt.contourf(xx,yy,Z,cmap="Pastel1")
plt.colorbar()
plt.scatter(X_train[Y_train==1,0],X_train[Y_train==1,1],s=5,marker="^",color="orange",label="training data feat. 0")
plt.scatter(X_train[Y_train==0,0],X_train[Y_train==0,1],s=5,marker="+",color="cornflowerblue",label="training data feat. 1")
plt.scatter(X_test[Y_test==1,0],X_test[Y_test==1,1],s=5,marker="^",color="red",label="training data feat. 0",alpha=0.5)
plt.scatter(X_test[Y_test==0,0],X_test[Y_test==0,1],s=5,marker="+",color="blue",label="training data feat. 1",alpha=0.5)
plt.legend()