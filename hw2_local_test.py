import pandas as pd

# Read data set and handle missing values
df = pd.read_csv('CreditCard.csv')
df = df.dropna(subset=['CreditApprove', 'Gender', 'CarOwner', 'PropertyOwner', '#Children', 'WorkPhone', 'Email_ID'])

# Encode binary attributes
df['Gender'] = df['Gender'].replace({'M': 1, 'F': 0})
df['CarOwner'] = df['CarOwner'].replace({'Y': 1, 'N': 0})
df['PropertyOwner'] = df['PropertyOwner'].replace({'Y': 1, 'N': 0})

# Initialize w-values
w = [1, 1, 1, 1, 1, 1]

# Number of records
num_records = 339

# Sum of squared errors
sum_squared_errors = 0

# Loop through each record in the data frame
for index, row in df.iterrows():
    y = row['CreditApprove']  # The actual result

    # Extract the attributes for dot product
    record = [
        row['Gender'],
        row['CarOwner'],
        row['PropertyOwner'],
        row['#Children'],
        row['WorkPhone'],
        row['Email_ID']
    ]

    # Compute dot product f(x) = w • record
    fx = sum(w[i] * record[i] for i in range(6))

    # Add squared error to sum
    sum_squared_errors += (fx - y) ** 2

# Calculate error
er = sum_squared_errors / num_records

print(er)
