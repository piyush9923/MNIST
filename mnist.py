import numpy as np
#import matplotlib.pyplot as plt
import tensorflow as tf

def initialize_parameters(n_x, n_h, n_y):
    W1 = np.random.randn(n_h, n_x)
    b1 = np.random.randn(n_h, 1)
    W2 = np.random.randn(n_y, n_h)
    b2 = np.random.randn(n_y, 1)
    
    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2}
    
    return parameters

#feedforward
def linear_activation_forward(A_prev, W, b):
    for bi, w in zip(b,W):
        A = sigmoid(np.dot(W,A_prev)+b)
    return A

#sigmoid
def sigmoid(z):
    return 1.0/(1.0+np.exp(-z))

#one-hot vector for labels
def onehot(label):
    E = np.zeros((10))
    E[label] = 1.0
    return E

#compute cost
def compute_cost(A2, Y, parameters):
    m = Y.shape[1]
    
    W1 = parameters['W1']
    W2 = parameters['W2']
    
    #cross entropy
    logprobs = np.multiply(np.log(A2), Y) + np.multiply((1-Y), np.log(1 - A2))
    cost = np.sum(logprobs)/m
    
    cost = np.squeeze(cost)
    
    assert(isinstance(cost, float))
    return cost

#back propogation
def back_propogation(parameters, A1, A2, X, Y):
    m = X.shape[0]
    print(m)
    W1 = parameters['W1']
    W2 = parameters['W2']
    
    dZ2 = A2 - Y
    dW2 = (1/m)* np.dot(dZ2, A1.T)
    
    print(dW2.shape)
    
    db2 = (1/m) * np.sum(dZ2, axis=1, keepdims=True)
    dZ1 = np.multiply(np.dot(W2.T,dZ2), 1 - np.power(A1,2))
    dW1 = (1/m) * np.dot(dZ1, X)
    db1 = (1/m) * np.sum(dZ1, axis=1, keepdims=True)
    
    grads = {"dW1": dW1,
            "db1": db1,
            "dW2": dW2,
            "db2": db2}
    
    return grads


#update parameters
def update_parameters(parameters, grads, learning_rate=0.25):
    
    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']
    
    dW1 = grads['dW1']
    db1 = grads['db1']
    dW2 = grads['dW2']
    db2 = grads['db2']
    
    W1 = W1 - learning_rate * dW1
    b1 = b1 - learning_rate * db1
    W2 = W2 - learning_rate * dW2
    b2 = b2 - learning_rate * db2
    
    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2}
    
    return parameters


#reducing the dataset
def redu(labels, data):
    a = np.zeros(shape=(10000,), dtype=np.int32)
    x = np.zeros(shape=(10000,784))
    b,c,d,e,f,g,h,j,k,l = 0,0,0,0,0,0,0,0,0,0
    
    for i in range(55000):
        sum = b+c+d+e+f+g+h+j+k+l
        if(labels[i]==0 and b<1000):
            a[sum] = labels[i]
            x[sum] = data[i]
            b += 1
        if(labels[i]==1 and c<1000):
            a[sum] = labels[i]
            x[sum] = data[i]
            c += 1
        if(labels[i]==2 and d<1000):
            a[sum] = labels[i]
            x[sum] = data[i]
            d += 1
        if(labels[i]==3 and e<1000):
            a[sum] = labels[i]
            x[sum] = data[i]
            e += 1
        if(labels[i]==4 and f<1000):
            a[sum] = labels[i]
            x[sum] = data[i]
            f += 1
        if(labels[i]==5 and g<1000):
            a[sum] = labels[i]
            x[sum] = data[i]
            g += 1
        if(labels[i]==6 and h<1000):
            a[sum] = labels[i]
            x[sum] = data[i]
            h += 1
        if(labels[i]==7 and j<1000):
            a[sum] = labels[i]
            x[sum] = data[i]
            j += 1
        if(labels[i]==8 and k<1000):
            a[sum] = labels[i]
            x[sum] = data[i]
            k += 1
        if(labels[i]==9 and l<1000):
            a[sum] = labels[i]
            x[sum] = data[i]
            l += 1
            
    return x,a


#accuracy
def accuracy(data, parameters, label ):
    index = np.argmax(Y, axis=0)
    correct = 0
    for i in range(10000):
        if (index[i] == label[i]):
            correct += 1
    
    acc = correct/10000
    print("Accuracy : ",str(acc*100))
    
    
if __name__ == "__main__":
    #load mnist
    learn = tf.contrib.learn
    tf.logging.set_verbosity(tf.logging.ERROR)
    mnist = learn.datasets.load_dataset('mnist')
    data1 = mnist.train.images
    labels1 = np.asarray(mnist.train.labels, dtype=np.int32)
    test_data = mnist.test.images
    test_labels = np.asarray(mnist.test.labels, dtype=np.int32)
    print(labels1.shape)
    
    #reduced dataset
    data, labels = redu(labels1, data1)
    
    one_hot_vectors = [onehot(y) for y in labels]
    one_hot_vectors = np.asarray(one_hot_vectors).T
    print(one_hot_vectors.shape)
    
    parameters = initialize_parameters(784,300,10)
    cost = 0
    for i in range(100):
        datat = data.T
        print("-------------------------------------")
        print('iteration = ', str(i+1))
        
        X = linear_activation_forward(datat, parameters['W1'], parameters['b1'])
        
        Y = linear_activation_forward(X, parameters['W2'], parameters['b2'])
        
        index = np.argmax(Y, axis=0)
        
        cost = compute_cost(Y, one_hot_vectors, parameters)
        print("Cost = ", str(cost))
        
        grads = back_propogation(parameters, X, Y, data, one_hot_vectors)
        
        parameters = update_parameters(parameters, grads)
        
        accuracy(Y, parameters, labels)
        
    
    print(Y.shape)
    accuracy(data,parameters,labels)