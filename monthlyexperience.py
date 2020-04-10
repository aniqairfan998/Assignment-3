
# Importing the libraries and the dataset for evaluation
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.tree import DecisionTreeRegressor

# Importing the dataset
df_raw = pd.read_csv('monthlyexp vs incom.csv')
a = df_raw.iloc[:, 0:1].values
b = df_raw.iloc[:, 1:2].values

# Splitting the dataset into the Training set and Test set
"""from sklearn.cross_validation import train_test_split
a_train, a_test, b_train, b_test = train_test_split(a, b, test_size = 0.2, random_state = 0)"""

# Fitting Linear Regression

lin_reg = LinearRegression()
lin_reg.fit(a, b)

# Fitting Polynomial Regression

poly_reg = PolynomialFeatures(degree = 4)
a_poly = poly_reg.fit_transform(a)
poly_reg.fit(a_poly, b)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(a_poly, b)

# Fitting Decision Tree Regression

regressor = DecisionTreeRegressor(random_state = 0)
regressor.fit(a, b)

# Visualising the Linear Regression results
plt.scatter(a, b, color = '#00CED1') 
plt.plot(a, lin_reg.predict(a), color = 'grey') 
plt.title('Linear Regression for Monthly Experience vs Income Distribution',fontweight='bold', fontname='Lucida Handwriting', fontsize=10 , color = 'grey')
plt.xlabel('Monthly Experience',fontweight='bold', fontname='Lucida Handwriting', fontsize=14 , color = '#00CED1') 
plt.ylabel('Income',fontweight='bold', fontname='Lucida Handwriting', fontsize=14 , color = '#00CED1') 
plt.grid(color='grey', linestyle='-', linewidth=0.5, alpha=0.5) 
plt.show()

# Visualising the Polynomial Regression results
a_grid = np.arange(min(a), max(a), 0.1)
a_grid = a_grid.reshape((len(a_grid), 1))
plt.scatter(a, b, color = '#00CED1')
plt.plot(a_grid, lin_reg_2.predict(poly_reg.fit_transform(a_grid)), color = 'grey') 

plt.title('Polynomial Regression for Monthly Experience vs Income Distribution',fontweight='bold', fontname='Lucida Handwriting', fontsize=10 , color = 'grey') 
plt.xlabel('Monthly Experience',fontweight='bold', fontname='Lucida Handwriting', fontsize=14 , color = '#00CED1') 
plt.ylabel('Income ',fontweight='bold', fontname='Lucida Handwriting', fontsize=14 , color = '#00CED1')
plt.grid(color='grey', linestyle='-', linewidth=0.5, alpha=0.5) 
plt.show()

# Visualising the Decision Tree Regression results
a_grid = np.arange(min(a), max(b), 0.01)
a_grid = a_grid.reshape((len(a_grid), 1))
plt.scatter(a, b, color = '#00CED1') 
plt.plot(a_grid, regressor.predict(a_grid), color = 'grey') 

plt.title('Decision Tree Regression for Monthly Experience vs. Income Distribution',fontweight='bold', fontname='Lucida Handwriting', fontsize=10 , color = 'grey')
plt.xlabel('Monthly Experience',fontweight='bold', fontname='Lucida Handwriting', fontsize=14 , color = '#00CED1')
plt.ylabel('Income ',fontweight='bold', fontname='Lucida Handwriting', fontsize=14 , color = '#00CED1') 
plt.grid(color='grey', linestyle='-', linewidth=0.5, alpha=0.5) 

plt.show()