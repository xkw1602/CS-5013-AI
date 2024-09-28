import pandas as pd

# Read in data and handle missing attributes
df = pd.read_csv('CreditCard.csv')
df = df.dropna(subset=['CreditApprove', 'Gender', 'CarOwner', 'PropertyOwner', '#Children', 'WorkPhone', 'Email_ID'])

# Define the mapping for each binary attribute
mapping = {
    'Gender': {'M': 1, 'F': 0},  
    'CarOwner': {'Y': 1, 'N': 0}, 
    'PropertyOwner': {'Y': 1, 'N': 0},
}

# Function to encode the DataFrame
def encode_binary_attributes(row):
    return [
        mapping['Gender'][row['Gender']],
        mapping['CarOwner'][row['CarOwner']],
        mapping['PropertyOwner'][row['PropertyOwner']],
    ]

# Apply the encoding to each row and create a new DataFrame
encoded_records = df.apply(encode_binary_attributes, axis=1)

# Convert the result into a DataFrame for easier manipulation
encoded_df = pd.DataFrame(encoded_records.tolist(), columns=['CreditApprove', 'Gender', 'CarOwner', 'PropertyOwner', '#Children', 'WorkPhone', 'Email_ID'])

# Print the encoded DataFrame
print(encoded_df)
