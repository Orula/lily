import time
import numpy as np
from MNIST_loader import mnist_dataset

#### Miscellaneous functions
class ANN:
  def __init__(self, size):
    self.params, self.cache, self.grads = {}, {}, {}
    self.size, self.num = size, len(size)

    self.params["b"] = [ np.random.randn(y, 1)*0.01 for y in size[1:] ]
    self.params["w"] = [ np.random.randn(y, x)*0.01 for x, y in zip(size[:-1], size[1:]) ]

    # gradients for learning
    self.grads["b"]  = [ np.zeros(b.shape) for b in self.params['b'] ]
    self.grads["w"]  = [ np.zeros(w.shape) for w in self.params['w'] ]


  def actFunc(self, z, deriv=False):
    # sigmoid
    if deriv: return (np.exp(-z))/((np.exp(-z)+1)**2)
    return 1/(1 + np.exp(-z))

    ## relu - not working
    #if deriv: return 1. * (z > 0)
    #return np.maximum(z, 0)


 
  def softmax(self, z, derivative=False):
    # Numerically stable with large exponentials
    exps = np.exp(z - z.max())
    if derivative:
        return exps / np.sum(exps, axis=0) * (1 - exps / np.sum(exps, axis=0))
    return exps / np.sum(exps, axis=0)

  def forward(self, image):
    for w, b in zip(self.params["w"], self.params["b"]):
      #image = self.actFunc( np.matmul(w, image) + b )
      image = self.actFunc( np.matmul(w, image) )
    #return self.softmax( image )
    return image

  def backward(self, image, label):
    layer_vec = []     # each layer's z vector
    layer_act = []     # each layer's z vector after activation function
      
    # determinig the output of each layer
    layer_act.append( image )     # adding the input to update first layer
    for w, b in zip(self.params["w"], self.params["b"]):
      image = np.matmul(w, image) + b
      layer_vec.append(image)

      image = self.actFunc(image) 
      layer_act.append(image)

    dout = image - label
    loss = image - label

    delta = dout * self.actFunc(layer_vec[-1], deriv=True)  
    self.grads["b"][-1] += delta
    self.grads["w"][-1] += np.dot(delta, layer_act[-2].T)
    # determing HIDDEN neurons weights and bias
    for l in range(2, self.num):
      delta = np.matmul(self.params["w"][-l+1].T, delta) * self.actFunc(layer_vec[-l], deriv=True)  #  delta * dz/da * input 
      self.grads["b"][-l] += delta
      self.grads["w"][-l] += np.matmul(delta, layer_act[-l-1].T)
    return loss 

  def optimize(self, batch_size, lr=0.8):
    #### SGD update rule
    self.params["b"] = [b-(lr/batch_size) * nb for b, nb in zip(self.params["b"], self.grads["b"])]
    self.params["w"] = [w-(lr/batch_size) * nw for w, nw in zip(self.params["w"], self.grads["w"])]

    ### clean grads
    self.grads["b"] = [ np.zeros(b.shape) for b in self.params['b'] ]
    self.grads["w"] = [ np.zeros(w.shape) for w in self.params['w'] ]

net = ANN(size=[784, 128, 64, 10])

x_train, y_train, x_test, y_test = mnist_dataset()

x_train = x_train.reshape((len(x_train),784,1))
x_test  = x_test.reshape((len(x_test),784,1))

y_train = y_train.reshape((len(y_train),10,1))
y_test  = y_test.reshape((len(y_test),10,1))


for epochs in range(1):
  #for img in x_train: print( np.argmax( net.forward( img ))   )
  #for img in x_train: print(  net.forward( img ).shape   )

  # the actuall learning portion
  for x, y in zip(x_train, y_train):
    net.backward(x, y)  # generate delta weights
    net.optimize(1)       # update weights with SGD

  results = [(np.argmax(net.forward(x)), np.argmax(y)) for (x, y) in zip(x_train, y_train)]
  tmp =  sum(int(x == y) for (x, y) in results)
  #print( tmp, len(results), tmp/len(results),)
  print( tmp/len(results) )
