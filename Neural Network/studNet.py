import numpy as np

def activation(z):
    '''
    ReLU function on a vector this is included for use as your activation function
    
    Input: a vector of elements to preform sigmoid on
    
    Output: a vector of elements with sigmoid preformed on them
    '''
    
    #return 1 / (1 + np.exp(-z))
    return np.maximum(0,z)


def deriv(z):
    '''
    The derivative of ReLU, you will need this to preform back prop
    '''
    z = (z>0).astype(int)
    z = np.where(z==0, 0.5, z) 
    
    #map(lambda x: x if x >0 else 0, z)
    #print(z)
    #return activation(z) * (1 - activation(z))
    
    return z 

class NeuralNetwork(object):     
    '''
    This Object outlines a basic neuralnetwork and the methods that it will need to function
    
    We have included an init method with a size parameter:
        Size: A 1D array indicating the node size of each layer
            E.G. Size = [2, 4, 1] Will instantiate weights and biases for a network
            with 2 input nodes, 1 hidden layer with 4 nodes, and an output layer with 1 node
        
        test_train defines the sizes of the input and output layers, but the rest is up to your implementation
    
    In this network for simplicity all nodes in a layer are connected to all nodes in the next layer, and the weights and
    biases and intialized as such. E.G. In a [2, 4, 1] network each of the 4 nodes in the inner layer will have 2 weight values
    and one biases value.
    
    '''

    def __init__(self, size, seed=73):
        #24  #73
        '''
        Here the weights and biases specified above will be instantiated to random values
        Your network will change these values to fit a certain dataset by training
        '''
        self.seed = seed
        np.random.seed(self.seed)
        self.size = size
        self.weights = [np.random.randn(self.size[i], self.size[i-1]) * np.sqrt(1 / self.size[i-1]) for i in range(1, len(self.size))]
        self.biases = [np.random.rand(n, 1) for n in self.size[1:]]
        
    def forward(self, input):
        '''
        Perform a feed forward computation 
        Parameters:

        input: data to be fed to the network with (shape described in spec)

        returns:

        the output value(s) of each example as ‘a’

        The values before activation was applied after the input was weighted as ‘pre_activations’

        The values after activation for all layers as ‘activations’

        You will need ‘pre_activaitons’ and ‘activations’ for updating the network
        '''
        
        a = input
        pre_activations = []
        activations = [a]
        weights = [self.weights[0]]
        biases = [self.biases[0]]
        for w, b in zip(weights, biases):
            z = np.dot(w, a) + b
            a  = activation(z)
            pre_activations.append(z)
            activations.append(a)
            
        weights = [self.weights[1]]
        biases = [self.biases[1]]   
        for w, b in zip(self.weights[1], self.biases[1]):
            z = np.dot(w, activations[1]) + b
            a  = activation(z)
            pre_activations.append(z)
            activations.append(a)       
        
        return a, pre_activations, activations
    

    def calcDeltas(self, activations,y):
        error_out = ((1 / 2) * (np.power((activations[2] - y), 2)))
 
        return error_out.sum()
    def backpropagate(self, pre_activations,activations,X,y,lr,g):
        cost_aout = activations[2] - y
        #cost_aout = (activations[2]-y)/activations[2]
        aout_zout = deriv(pre_activations[1])
        zout_wout = activations[1]
        
        #print(aout_zout)
        
        cost_wout = np.dot(zout_wout, (cost_aout * aout_zout).T)

        
        zh_wh = X
        ah_zh =  deriv(pre_activations[0])
        cost_ah = np.dot((aout_zout*cost_aout).T, self.weights[1]  )
        cost_wh1 = np.dot(zh_wh, cost_ah*ah_zh.T   )
        
        #print(cost_wh1)
        
        #update weight
        self.weights[0] = self.weights[0] - lr * cost_wh1.T
        self.weights[1] = self.weights[1] - lr * cost_wout.T
        #print(self.weights)
        
        #update biases
        delta = (cost_aout*aout_zout).T
        self.biases[1] -= g * delta.sum()
        delta =  cost_ah*ah_zh.T   
        self.biases[0] -= g * delta.sum()
             
        return None
    def train(self, X, y):
        lr = 0.0002
        #lr = 0.02
        g = 0.00002
        #print('biases: ', self.biases)
        #print('at start: ',self.weights)
        for i in range (4000):
            #print('Iteration: ',i)
            a, pre_activations, activations = self.forward(X)
            #print('error: ',self.calcDeltas( activations,y))
            self.backpropagate(pre_activations,activations,X,y,lr,g)
            #print('error: ',self.calcDeltas( activations,y)) 
        #print('error: ',self.calcDeltas( activations,y))    
        return None

    def predict(self, a):
        '''
       Input: a: list of list of input vectors to be tested
       
       This method will test a vector of input parameter vectors of the same form as X in test_train
       and return the results (Zero or One) that your trained network came up with for every element.
       
       This method does this the same way the included forward method moves an input through the network
       but without storying the previous values (which forward stores for use with the delta function you must write)
        '''
        for w, b in zip(self.weights, self.biases):
            z = np.dot(w, a) + b
            a = activation(z)
        predictions = (a > 0.5).astype(int)
        return predictions
    
'''
This is the function that we will call to test your network.

It must instantiate a network, which we include.

It must then train the network given the passed data, where x is the parameters in form:
        [[1rst parameter], [2nd parameter], [nth parameter]]
    
        Where if there are 100 training examples each of the n lists inside the list above will have 100 elements
        
    Y is the target which is guarenteed to be binary, or in other words true or false:
    Y will be of the form: 
        [[1, 0, 0, ...., 1, 0, 1]]
        
        (where 1 indicates true and zero indicates false)

'''
def test_train(X, y):
    inputSize = np.size(X, 0)
    
    #feel free to change the inside (hidden) layers to best suite your implementation
    #but the sizes of the input layer and output layer (inputSize and 1) must NOT CHANGE
    #start with 1 hidden layer first
    retNN = NeuralNetwork([inputSize, 7, 1])
    #train your network here
    retNN.train(X, y)
    
    #then the function MUST return your TRAINED nueral network
    return retNN
    