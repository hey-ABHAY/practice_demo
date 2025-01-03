import pandas as pd
import numpy as np
import matplotlib.pyplot as plt # type: ignore

df=pd.read_csv('/Users/abhaysingh/Library/Mobile Documents/com~apple~Numbers/Documents/cars 2.csv')

df.drop(['Car','Model'],axis=1,inplace=True)
df.head()
x=df[['Volume','Weight']]
y=df['CO2']
from sklearn.preprocessing import StandardScaler # type: ignore
sc=StandardScaler()
x=sc.fit_transform(x)
from sklearn.model_selection import train_test_split # type: ignore
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
print(x_train)

