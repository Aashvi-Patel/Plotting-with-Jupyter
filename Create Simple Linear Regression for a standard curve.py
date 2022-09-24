# Importing the libraries
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from matplotlib import pyplot as plt

#Import the CSV file into the dataframe
data = pd.read_csv(r'fastgreendata.csv')

#Test if correct data has been loaded (uncomment line 11)
#print(data)

#Define x and y values
x = data['Final Fast Green Concentration']
y = data['OD_635 avg']

#Training the Simple Linear Regression model
linreg = LinearRegression()
x = x.values.reshape(-1,1)
linreg.fit(x,y)
y.predict = linreg.predict(x)

#Visualizing data
plt.scatter(x,y)
plt.plot(x,y.predict, color='teal')

#Find slope, y-intercept, and R squared value
slope = linreg.coef_
intercept = linreg.intercept_
Rsqr = linreg.score(x,y)

#Set titles and axis labels
plt.title("Fast Green Standard Curve")
plt.xlabel("Final Fast Green Concentration")
plt.ylabel("OD_635 avg")

#Add grid to graph
plt.grid

#Print equation of the line and the R squared value
print("y =",slope,"x","+",intercept)
print("R^2=",Rsqr)