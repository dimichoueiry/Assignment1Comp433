import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('Copy of lung cancer dataset.csv')

print(data.head())

print(data["AGE"])

print("This is the data for gender: ", data["AGE"].where(data["AGE"] > 21))

print(data["AGE"].shape)

print(data.columns)

for i in data.columns:
    print(i)
    
    
print("This is the max", data["AGE"].max())