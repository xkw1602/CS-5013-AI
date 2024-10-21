import numpy as np
import matplotlib.pyplot as plt

# --- Your Task --- #
# import necessary library 
# if you need more libraries, just import them
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
# --- end of task --- #

# load a data set for regression
# in array "data", each row represents a community 
# each column represents an attribute of community 
# last column is the continuous label of crime rate in the community
data = np.loadtxt('Homework 4\crimerate.csv', delimiter=',', skiprows=1)
[n,p] = np.shape(data)

# always use last 25% data for testing 
num_test = int(0.25*n)
sample_test = data[n-num_test:,0:-1]
label_test = data[n-num_test:,-1]

# --- Your Task --- #
# now, pick the percentage of data used for training 
# remember we should be able to observe overfitting with this pick 
# note: maximum percentage is 0.75 
per = 0.675
num_train = int(n*per)
sample_train = data[0:num_train,0:-1]
label_train = data[0:num_train,-1]
# --- end of task --- #


# --- Your Task --- #
# We will use a regression model called Ridge. 
# This model has a hyper-parameter alpha. Larger alpha means simpler model. 
# Pick 5 candidate values for alpha (in ascending order)
# Remember we should aim to observe both overfitting and underfitting from these values 
# Suggestion: the first value should be very small and the last should be large 
alpha_vec = [0.05, 0.1, 0.5, 1.0, 5.0]
# --- end of task --- #

er_train_alpha = []
er_test_alpha = []

er_valid_alpha = []
k = 5

indices = np.arange(num_train)
folds = np.array_split(indices, k)
    
for alpha in alpha_vec: 

    # pick ridge model, set its hyperparameter 
    model = Ridge(alpha = alpha)
    
    # --- Your Task --- #

    # now implement k-fold cross validation 
    # on the training set (which means splitting 
    # training set into k-folds) to get the 
    # validation error for each candidate alpha value 
    # store it in "er_valid"

    fold_errors = []

    for i in range(k):
        valid_indices = folds[i]
        train_indices = []

        for j in range(k):
            if j != i:
                train_indices.append(folds[j])

        train_indices = np.concatenate(train_indices)
        
        sample_train_fold = sample_train[train_indices]
        label_train_fold = label_train[train_indices]
        sample_valid_fold = sample_train[valid_indices]
        label_valid_fold = label_train[valid_indices]

        model.fit(sample_train_fold, label_train_fold)

        predictions = model.predict(sample_valid_fold)

        fold_error = mean_squared_error(label_valid_fold, predictions)
        fold_errors.append(fold_error)
    # ......
    # ......
    er_valid = np.mean(fold_errors)
    er_valid_alpha.append(er_valid)
    # --- end of task --- #


# Now you should have obtained a validation error for each alpha value 
# In the homework, you just need to report these values
print(f'Alpha Values: {alpha_vec}\nErrors:{er_valid_alpha}')
# The following practice is only for your own learning purpose.
# Compare the candidate values and pick the alpha that gives the smallest error 
# set it to "alpha_opt"
alpha_opt = ...

# now retrain your model on the entire training set using alpha_opt 
# then evaluate your model on the testing set 
model = Ridge(alpha = alpha_opt)
# ......
# ......
# ......
er_train = ...
er_test = ...


