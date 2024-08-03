import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
df = pd.read_csv('data/preprocessed_creditcard.csv')

# Plot
plt.figure(figsize=(10, 6))
sns.histplot(df['Amount'], kde=True)
plt.title('Distribution of Transaction Amounts')
plt.show()
