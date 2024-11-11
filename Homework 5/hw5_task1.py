#
# Template for Task 1: Linear Regression 
#
import numpy as np
import matplotlib.pyplot as plt

# --- Your Task --- #
# import libraries as needed 
# .......
# --- end of task --- #

# -------------------------------------
# load data 
data = np.loadtxt('crimerate.csv', delimiter=',')
[n,p] = np.shape(data)
# 75% for training, 25% for testing 
num_train = int(0.75*n)
num_test = int(0.25*n)
sample_train = data[0:num_train,0:-1]
label_train = data[0:num_train,-1]
sample_test = data[n-num_test:,0:-1]
label_test = data[n-num_test:,-1]
# -------------------------------------


# --- Your Task --- #
# pick a proper number of iterations 
num_iter = 100
# randomly initialize your w 
w = np.random.rand(sample_train.shape[1])
# --- end of task --- #

er_test = []


# --- Your Task --- #
# implement the iterative learning algorithm for w
# at the end of each iteration, evaluate the updated w 
alpha = 0.01

for iter in range(num_iter): 
    # Calculate predictions for the training set
    predictions = sample_train.dot(w)
    
    # Calculate the error
    errors = predictions - label_train
    
    # Calculate the gradient
    gradient = (1 / num_train) * sample_train.T.dot(errors)
    
    # Update weights
    w -= alpha * gradient

    # Calculate MSE for the test set
    test_predictions = sample_test.dot(w)
    er = np.mean((test_predictions - label_test) ** 2)
    er_test.append(er)
# --- end of task --- #
    
plt.figure()    
plt.plot(er_test)
plt.xlabel('Iteration')
plt.ylabel('Classification Error')
plt.show()


