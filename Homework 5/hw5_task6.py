#
# Template for Task 6: Random Forest Classification 
#
import numpy as np
import matplotlib.pyplot as plt

# --- Your Task --- #
# import libraries as needed 
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
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
# pick five values of m by yourself 
m_values = [1, 10, 100, 1000, 10000]
# --- end of task --- #

er_test = []
for m in m_values: 
    rf_classifier = RandomForestClassifier(n_estimators=m, random_state=0)
    
    # Train the classifier on the training data
    rf_classifier.fit(sample_train, label_train)
    
    # Predict on the test data
    predictions = rf_classifier.predict(sample_test)
    
    # Calculate classification error for the current m
    accuracy = accuracy_score(label_test, predictions)
    er = 1 - accuracy
    er_test.append(er)
# --- end of task --- #
    
plt.figure()    
plt.plot(er_test)
plt.xlabel('m')
plt.ylabel('Classification Error')
plt.show()


