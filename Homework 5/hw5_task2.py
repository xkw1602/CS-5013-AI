#
# Template for Task 2: Logistic Regression 
#
import numpy as np
import matplotlib.pyplot as plt

# --- Your Task --- #
# import libraries as needed 
# .......
# --- end of task --- #

# -------------------------------------
# load data 
data = np.loadtxt('diabetes.csv', delimiter=',')
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
num_iter = 1000
# randomly initialize your w 
w = np.random.rand(sample_train.shape[1])
# --- end of task --- #

er_test = []


# --- Your Task --- #
# implement the iterative learning algorithm for w
# at the end of each iteration, evaluate the updated w 
alpha = 0.01

def logistic(z):
    return 1 / (1 + np.exp(-z))

for iter in range(num_iter): 

    ## update w

    # Calculate predictions for the training set 
    predictions = logistic(sample_train.dot(w))
    
    # Calculate the error 
    errors = predictions - label_train
    
    # Calculate the gradient
    gradient = (1 / num_train) * sample_train.T.dot(errors)
    
    # Update weights
    w -= alpha * gradient

    ## evaluate testing error of the updated w 
    # Calculate classification error for the test set
    test_predictions = logistic(sample_test.dot(w))
    # Convert probabilities to binary predictions (0 or 1) by thresholding at 0.5
    test_predictions_binary = (test_predictions >= 0.5).astype(int)
    # Calculate classification error as the fraction of incorrect predictions
    er = np.mean(test_predictions_binary != label_test)
    er_test.append(er)
# --- end of task --- #
    
plt.figure()    
plt.plot(er_test)
plt.xlabel('Iteration')
plt.ylabel('Classification Error')
plt.show()


