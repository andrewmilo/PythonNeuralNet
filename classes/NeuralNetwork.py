import numpy as np

class Neural_Network(object):
  def __init__(self):
    self.input_neuron_count = 2
    self.inner_neuron_count = 3
    self.output_neuron_count = 1
    self.W1 = np.random.randn(self.input_neuron_count, self.inner_neuron_count) 
    self.W2 = np.random.randn(self.inner_neuron_count, self.output_neuron_count)

  #forward propagation through our network
  def forward(self, X):
    self.z = np.dot(X, self.W1) # dot product of X (input) and first set of 3x2 weights
    self.z2 = self.sigmoid(self.z) # activation function
    self.z3 = np.dot(self.z2, self.W2) # dot product of hidden layer (z2) and second set of 3x1 weights
    o = self.sigmoid(self.z3) # final activation function
    return o

  def sigmoid(self, s):
    # activation function
    return 1/(1+np.exp(-s))

  def sigmoidPrime(self, s):
    #derivative of sigmoid
    return s * (1 - s)

  def backward(self, X, y, o):
    # backward propagate through the network
    self.o_error = y - o # error in output
    self.o_delta = self.o_error*self.sigmoidPrime(o) # applying derivative of sigmoid to error

    self.z2_error = self.o_delta.dot(self.W2.T) # z2 error: how much our hidden layer weights contributed to output error
    self.z2_delta = self.z2_error*self.sigmoidPrime(self.z2) # applying derivative of sigmoid to z2 error

    self.W1 += X.T.dot(self.z2_delta) # adjusting first set (input --> hidden) weights
    self.W2 += self.z2.T.dot(self.o_delta) # adjusting second set (hidden --> output) weights

  def train (self, X, y):
    o = self.forward(X)
    self.backward(X, y, o)

  def predict(self):
    print "Predicted data based on trained weights: ";
    print "Input (scaled): \n" + str(xPredicted);
    print "Output: \n" + str(self.forward(xPredicted));