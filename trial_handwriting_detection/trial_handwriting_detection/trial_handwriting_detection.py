import numpy as np
import pickle as cPickle
import gzip
import random

def load_data():
    f=gzip.open('../mnist.pkl.gz', 'rb')
    training_data, validation_data, test_data = cPickle.load(f, encoding="latin1")
    f.close()
    return (training_data, validation_data, test_data)
def load_data_wrapper():
    tr_d,va_d,te_d = load_data()
    print(tr_d)

    training_inputs=[]
    training_results=[]
    for x in tr_d[0]:
        training_inputs.append(np.reshape(x,(784,1)))
    for x in tr_d[1]:
        training_results.append(vectorized_result(x))
    training_data = zip(training_inputs,training_results)

    validation_inputs=[]
    for x in va_d[0]:
        validation_inputs.append(np.reshape(x,(784,1)))
    validation_data = zip(validation_inputs, va_d[1])

    test_inputs = []
    for x in te_d[0]:
        test_inputs.append(np.reshape(x, (784, 1)))
    test_data = zip(test_inputs, te_d[1])

    return (training_data, validation_data, test_data)

def vectorized_result(j):
    e = np.zeros((10, 1))
    e[j] = 1.0
    return e

class Network(object):
    def __init__(self,sizes):
        self.num_layers=len(sizes)
        self.sizes =sizes
        self.biases=[]
        self.weights=[]
        for y in sizes[1:]:
            self.biases.append(np.random.randn(y,1))
        for x,y in zip(sizes[:-1],sizes[1:]):
            self.weights.append(np.random.randn(y,x))

    def feedforward(self, a):
        for b, w in zip(self.biases, self.weights):
            a = self.sigmoid(np.dot(w, a)+b)
        return a
    
    def SGD(self, training_data, epochs, mini_batch_size, eta,test_data=None):
        test_data = list(test_data)
        training_data = list(training_data)
        if test_data: n_test = len(test_data)
        n = len(training_data)
        for j in range(epochs):
            random.shuffle(training_data)
            mini_batches = []
            for k in range(0, n, mini_batch_size):
                mini_batches.append( training_data[k:k+mini_batch_size])
            print(mini_batches)
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)
            if test_data:
                print ("Epoch {0}: {1} / {2}".format(j, self.evaluate(test_data), n_test))
            else:
                print ("Epoch {0} complete".format(j))

    def update_mini_batch(self, mini_batch, eta):
        nabla_b = []
        nabla_w = []
        for b in self.biases:
            nabla_b.append(np.zeros(b.shape))
        for w in self.weights:
            nabla_w.append(np.zeros(w.shape))
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        self.weights = [w-(eta/len(mini_batch))*nw for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b-(eta/len(mini_batch))*nb for b, nb in zip(self.biases, nabla_b)]
    def backprop(self, x, y):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        # feedforward
        activation = x
        activations = [x] # list to store all the activations, layer by layer
        zs = [] # list to store all the z vectors, layer by layer
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation)+b
            zs.append(z)
            activation = self.sigmoid(z)
            activations.append(activation)
        delta = self.cost_derivative(activations[-1], y) * \
            self.sigmoid_prime(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = self.sigmoid_prime(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
        return (nabla_b, nabla_w)

    def evaluate(self, test_data):
        test_results = [(np.argmax(self.feedforward(x)), y)
                        for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)

    def cost_derivative(self, output_activations, y):
        return (output_activations-y)

    def sigmoid(self,z):
        """The sigmoid function."""
        return 1.0/(1.0+np.exp(-z))

    def sigmoid_prime(self,z):
        """Derivative of the sigmoid function."""
        return self.sigmoid(z)*(1-self.sigmoid(z))

training_data, validation_data, test_data=load_data_wrapper()
net = Network([784, 30, 10])
net.SGD(training_data, 30, 10, 3.0, test_data=test_data)