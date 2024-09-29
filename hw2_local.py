import pandas as pd
import random
# Read data set and handle missing values
df = pd.read_csv('CreditCard.csv')
df = df.dropna(subset=['CreditApprove', 'Gender', 'CarOwner', 'PropertyOwner', '#Children', 'WorkPhone', 'Email_ID'])

# Encode binary attributes
df['Gender'] = df['Gender'].replace({'M':1, 'F':0})
df['CarOwner'] = df['CarOwner'].replace({'Y': 1, 'N' : 0})
df['PropertyOwner'] = df['PropertyOwner'].replace({'Y': 1, 'N': 0})


# randomly generate initial w-values
w = [random.choice([1, -1]) for _ in range(6)]
print(f'Initial w: {w}')

# initialize record for calculation
record = [-1, -1, -1, -1, -1, -1]

# Ignored one record due to missing value, so 339 records instead of 340
num_records = 339

# error calculation function
def calculate_error(w):
    sum_squared_errors = 0

    for index, row in df.iterrows():
        y = row['CreditApprove']
        record[0] = row['Gender']
        record[1] = row['CarOwner']
        record[2] = row['PropertyOwner']
        record[3] = row['#Children']
        record[4] = row['WorkPhone']
        record[5] = row['Email_ID']

        fx = 0
        for x in range(6):
            fx += w[x]*record[x]

        sum_squared_errors += (fx - y)**2
        
    er = sum_squared_errors/num_records
    return(er)

# function to generate neighbors of w 
def generate_neighbors(w):
    neighbors = []
    for i in range(len(w)):
        neighbor = w.copy()
        neighbor[i] = -1 * w[i]
        neighbors.append(neighbor)
    return(neighbors)

current_error = calculate_error(w)
print(f'Initial error: {current_error}')
iteration = 0

# Hill climbing loop
while True:
    iteration += 1
    neighbors = generate_neighbors(w)

    best_neighbor = w
    best_error = current_error

    # For all neighbors, calculate error and pick the lowest one
    for neighbor in neighbors:
        neighbor_error = calculate_error(neighbor)
        if(neighbor_error < best_error):
            best_neighbor = neighbor
            best_error = neighbor_error

    # No neighbors of current w produce a lower error, break loop
    if(best_error >= current_error):
        print('Done.')
        break

    w = best_neighbor
    current_error = best_error
    print(f'Iteration {iteration}: \n w = {w} \n error = {current_error}')
