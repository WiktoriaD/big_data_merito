import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


file = pd.read_csv("weight-height.csv", delimiter=';')
# print(file.head())
# print(file.describe())

# null_counts = file.isnull().sum()
# print("Null values:")
# print(null_counts.to_string())
# print(file['Gender'].nunique())
# print(file['Gender'].value_counts())
df = file.copy()
df['Height'] *= 2.54
df['Weight'] /= 2.2
print(df.head())

print('\nChart')
plt.hist(df['Height'], bins=20)
plt.xlabel('Height [cm]')
plt.show()