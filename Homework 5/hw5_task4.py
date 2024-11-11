#
# Template for Task 4: kNN Classification 
#
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
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
# pick five values of k by yourself 
k_values = [1, 3, 5, 7, 9]
# --- end of task --- #

# Function to calculate Euclidean distance
def euclidean_distance(a, b):
    return np.sqrt(np.sum((a - b) ** 2))

er_test = []
for k in k_values: 
    # --- Your Task --- #
    # List to store predictions for each test point
    predictions = []
    
    # Loop through each test sample
    for test_point in sample_test:
        # Calculate distances between the test point and all training points
        distances = [euclidean_distance(test_point, train_point) for train_point in sample_train]
        
        # Get the indices of the k nearest neighbors
        k_nearest_indices = np.argsort(distances)[:k]
        
        # Get the labels of the k nearest neighbors
        k_nearest_labels = [label_train[i] for i in k_nearest_indices]
        
        # Predict the label by taking the majority vote
        most_common_label = Counter(k_nearest_labels).most_common(1)[0][0]
        predictions.append(most_common_label)
    # store classification error on testing data here 
    er = np.mean(predictions != label_test)
    er_test.append(er)
# --- end of task --- #
    
plt.figure()    
plt.plot(k_values, er_test)
plt.xlabel('k')
plt.ylabel('Classification Error')
plt.show()


