import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression


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
# plt.hist(df['Weight'], bins=20)
# plt.xlabel('Weight [kg]')
# plt.show()

# plt.hist(df.query('Gender=="Male"')['Weight'], bins=50)
# plt.hist(df.query('Gender=="Female"')['Weight'], bins=50)
# plt.show()
#
# sns.histplot(df.query('Gender=="Male"')['Weight'])
# sns.histplot(df.query('Gender=="Female"')['Weight'])
# plt.show()

#zamiana gener na dane numeryczne
df = pd.get_dummies(df)
print(df.head())
del df['Gender_Male']
print(df.head())
df = df.rename(columns={'Gender_Female': 'Gender'})
print(df.head())
# False - man, True - woman
model = LinearRegression()
model.fit(df[['Height', 'Gender']], df['Weight'])
print(model.coef_) # współczynnik kierunkowy, czyli a ze wzoru y=ax+b
print(model.intercept_) # wyraz wolny, czyli b
print(f'Weight = Height * {model.coef_[0]} + Gender * {model.coef_[1]} + {model.intercept_}')
