# Import necessary libraries

# For data manipulation and analysis
import pandas as pd
# Documentation: https://pandas.pydata.org/docs/

# For creating visualizations
import matplotlib.pyplot as plt  
# Documentation: https://matplotlib.org/stable/contents.html

# For linear regression
from sklearn.linear_model import LinearRegression
# Documentation: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html

# Read the CSV file into a DataFrame
df = pd.read_csv('csvExamples/randomStudentData.csv')
# Documentation: https://pandas.pydata.org/docs/reference/api/pandas.read_csv.html

# Extract the independent variable (IQ) and dependent variable (Marks)
X = df[['IQ']]
y = df['Marks']

# Create a linear regression model
model = LinearRegression()

# Fit the model to the data
model.fit(X, y)

# Make predictions using the model
y_pred = model.predict(X)

# Plot the original data points
plt.scatter(X, y, color='blue', label='Original Data')  
# Documentation: https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.scatter.html

# Plot the regression line
plt.plot(X, y_pred, color='red', linewidth=2, label='Linear Regression')  
# Documentation: https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.plot.html

# Set the x-axis label
plt.xlabel('IQ')  
# Documentation: https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.xlabel.html

# Set the y-axis label
plt.ylabel('Marks')  
# Documentation: https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.ylabel.html

# Set the title of the plot
plt.title('Linear Regression: Marks vs IQ')  
# Documentation: https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.title.html

# Show the legend
plt.legend()  
# Documentation: https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.legend.html

# Display the plot
plt.show()  
# Documentation: https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.show.html
