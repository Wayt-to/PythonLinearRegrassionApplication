import pandas as pd
import numpy as np
# Step 1: Read the data
# Load the data from the Excel file
data = pd.read_excel(r"-EXCEL FILE PATH-")
# Removing the 'Home' column
data = data.drop(columns=['Home'])
print("Data Preview:")
print(data.head(10))
# Target variable (Price)
b = data['Price'].values
# Features (SqFt and Bedrooms)
A = data[['SqFt', 'Bedrooms']].values
# Adding a column of 1's for the intercept term
A = np.hstack([np.ones((A.shape[0], 1)), A])
print("\nFeature Matrix (A) and Target Vector (b):")
print("A (first 10 rows):")
print(A[:10])
print("b (first 10 values):")
print(b[:10])
# Step 3: Solve the linear regression system (A * v = b)
# I used the normal equation to find the coefficients (v)
v = np.linalg.inv(A.T @ A) @ A.T @ b
print("\nRegression Coefficients:")
print(f"Intercept (a0): {v[0]:.2f}")
print(f"Coefficient for SqFt (a1): {v[1]:.2f}")
print(f"Coefficient for Bedrooms (a2): {v[2]:.2f}")
# Step 4: Predict new values and apply the regression equation: y = a0 + a1*x1 + a2*x2
# I predict the price for a house with 1750 SqFt and 2 Bedrooms
new_sqft = 1750
new_bedrooms = 2
predicted_price = v[0] + v[1] * new_sqft + v[2] * new_bedrooms
print(f"\nPredicted Price for {new_sqft} SqFt and {new_bedrooms} Bedrooms:{predicted_price:.2f}")
