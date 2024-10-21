import numpy as np
import matplotlib.pyplot as plt

# --- Your Task --- #
# import necessary library 
# if you need more libraries, just import them
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.utils import resample
# --- end of task --- #

# load an imbalanced data set 
# there are 50 positive class instances 
# there are 500 negative class instances 
data = np.loadtxt('Homework 4/diabetes_new.csv', delimiter=',', skiprows=1)
[n,p] = np.shape(data)

# always use last 25% data for testing 
num_test = int(0.25*n)
sample_test = data[n-num_test:,0:-1]
label_test = data[n-num_test:,-1]

# vary the percentage of data for training
num_train_per = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8]

acc_base_per = []
auc_base_per = []

acc_yours_per = []
auc_yours_per = []

for per in num_train_per: 

    # create training data and label
    num_train = int(n*per)
    sample_train = data[0:num_train,0:-1]
    label_train = data[0:num_train,-1]

    model = LogisticRegression()

    # --- Your Task --- #
    # Implement a baseline method that standardly trains 
    # the model using sample_train and label_train
    model.fit(sample_train, label_train)
    # ......
    # ......
    
    # evaluate model testing accuracy and stores it in "acc_base"
    predictions_base = model.predict(sample_test)
    acc_base = accuracy_score(label_test, predictions_base)
    acc_base_per.append(acc_base)
    
    # evaluate model testing AUC score and stores it in "auc_base"
    probabilities_base = model.predict_proba(sample_test)[:, 1]
    auc_base = roc_auc_score(label_test, probabilities_base)
    auc_base_per.append(auc_base)
    # --- end of task --- #
    
    
    # --- Your Task --- #
    # Now, implement your method 
    # Aim to improve AUC score of baseline 
    # while maintaining accuracy as much as possible 

    # Random Oversampling
    majority_indices = np.where(label_train == 0)[0]
    minority_indices = np.where(label_train == 1)[0]

    oversampled_minority = resample(minority_indices, replace=True, n_samples=len(majority_indices))

    oversampled_indices = np.concatenate([majority_indices, oversampled_minority])

    sample_train_undersampled = sample_train[oversampled_indices]
    label_train_undersampled = label_train[oversampled_indices]

    model.fit(sample_train_undersampled, label_train_undersampled)
    # ......
    # ......
    # evaluate model testing accuracy and stores it in "acc_yours"
    predictions_yours = model.predict(sample_test)
    acc_yours = accuracy_score(label_test, predictions_yours)
    acc_yours_per.append(acc_yours)
    # evaluate model testing AUC score and stores it in "auc_yours"
    probabilities_yours = model.predict_proba(sample_test)[:, 1]
    auc_yours = roc_auc_score(label_test, probabilities_yours)
    auc_yours_per.append(auc_yours)
    # --- end of task --- #
    

plt.figure()    
plt.plot(num_train_per,acc_base_per, label='Base Accuracy')
plt.plot(num_train_per,acc_yours_per, label='Your Accuracy')
plt.xlabel('Percentage of Training Data')
plt.ylabel('Classification Accuracy')
plt.legend()


plt.figure()
plt.plot(num_train_per,auc_base_per, label='Base AUC Score')
plt.plot(num_train_per,auc_yours_per, label='Your AUC Score')
plt.xlabel('Percentage of Training Data')
plt.ylabel('Classification AUC Score')
plt.legend()
plt.show()    


