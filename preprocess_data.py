import pandas as pd
from sklearn.preprocessing import StandardScaler

# Load the dataset
df = pd.read_csv('data/creditcard.csv')

# Normalize 'Amount' and 'Time'
df['Amount'] = StandardScaler().fit_transform(df['Amount'].values.reshape(-1, 1))
df['Time'] = StandardScaler().fit_transform(df['Time'].values.reshape(-1, 1))

# Save the preprocessed data
df.to_csv('data/preprocessed_creditcard.csv', index=False)

print("Data preprocessing completed and saved to 'preprocessed_creditcard.csv'")
import pandas as pd

# Load the preprocessed dataset
df = pd.read_csv('data/preprocessed_creditcard.csv')

# Display basic information and statistics
print(df.info())
print(df.describe())
