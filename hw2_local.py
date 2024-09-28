import pandas as pd

# Read data set and handle missing values
df = pd.read_csv('CreditCard.csv')
df = df.dropna(subset=['CreditApprove', 'Gender', 'CarOwner', 'PropertyOwner', '#Children', 'WorkPhone', 'Email_ID'])

# Encode binary attributes
df['Gender'] = df['Gender'].replace({'M':1, 'F':0})
df['CarOwner'] = df['CarOwner'].replace({'Y': 1, 'N' : 0})
df['PropertyOwner'] = df['PropertyOwner'].replace({'Y': 1, 'N': 0})


# initialize w-values
w = [1, 1, 1, 1, 1, 1]

# initialize record for calculation
record = [-1, -1, -1, -1, -1, -1, -1]

for index, row in df.iterrows():
    record[0] = row['CreditApprove']
    record[1] = row['Gender']
    record[2] = row['CarOwner']
    record[3] = row['PropertyOwner']
    record[4] = row['#Children']
    record[5] = row['WorkPhone']
    record[6] = row['Email_ID']

