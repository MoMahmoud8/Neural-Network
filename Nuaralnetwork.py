



import numpy as np
from sklearn.metrics import mean_squared_error,accuracy_score


neuron = 4

  

def sigmoid_Function(x):
    return 1.0/(1+ np.exp(-x))

def sigmoid_derivative_Function(x):
    return x * (1.0 - x)

class Neural_Network:
    def __init__(self, x, y):
        self.input = x
    #    print(" input-->",self.input.shape)
   
        self.weight1   = np.random.rand(self.input.shape[1],neuron) 
    #    print("weights1-->",self.weight1.shape)
    
        self.weight2   = np.random.rand(neuron,1)                 
    #    print("weights2-->",self.weight2.shape)
    
        self.y  = y
    #    print("y-->",self.y.shape)
    
        self.output = np.zeros(self.y.shape) # y hat
    #    print("output-->",self.output.shape)
        
    def forward_propagation(self):
        self.First_layer = sigmoid_Function(np.dot(self.input, self.weight1))
    
        self.output = sigmoid_Function(np.dot(self.First_layer, self.weight2))

        
    def back_propagation(self):
        delta=(self.y-self.output)*sigmoid_derivative_Function(self.output)
        
        w_weight2 = np.dot(self.First_layer.T, delta) #change at output layer
           
        w_weight1 = np.dot(self.input.T,
                          (np.dot(delta , self.weight2.T) * sigmoid_derivative_Function(self.First_layer)))    #at hidden layer       

        
        # update the weights 
        self.weight1 += w_weight1
        self.weight2 += w_weight2




X= np.array( [[0,0,0,1],
              [0,0,1,1],
              [0,1,0,1],
              [0,1,1,1],
              [1,0,0,1],
              [1,0,1,1],
              [1,1,0,1],
             [1,1,1,1]])              
## shape--> 8 * 4



y = np.array([[0],
              [1],
              [1],
              [1],
              [1],
              [1],
              [1],
             [0]])

##shape--> 8 * 1
my_nn = Neural_Network(X,y)

for i in range(100):
    my_nn.forward_propagation()
    my_nn.back_propagation()

#print(my_nn.output)

#print(accuracy_score(y,my_nn.output))
print("MSE--> ",mean_squared_error(y,my_nn.output))
