import numpy as np
from classes import NeuralNetwork

nn = NeuralNetwork.Neural_Network()

# X = (hours sleeping, hours studying), y = score on test
X = np.array(([2, 9], [1, 5], [3, 6]), dtype=float)
y = np.array(([92], [86], [89]), dtype=float)

# scale units
X = X/np.amax(X, axis=0) # maximum of X array
y = y/100 # max test score is 100

xPredicted = np.array(([4,8]), dtype=float)
xPredicted = xPredicted/np.amax(xPredicted, axis=0) # maximum of xPredicted (our input data for the prediction)

for i in xrange(120): # trains the NN 1,000 times
  print "Input: \n" + str(X)
  print "Actual Output: \n" + str(y)
  print "Predicted Output: \n" + str(nn.forward(X))
  print "Loss: \n" + str(np.mean(np.square(y - nn.forward(X)))) # mean sum squared loss
  print "\n"
  nn.train(X, y)
  nn.predict()