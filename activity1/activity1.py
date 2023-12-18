# Import necessary libraries

# For data manipulation and analysis
import pandas as pd
# Documentation: https://pandas.pydata.org/docs/

# For creating visualizations
import matplotlib.pyplot as plt  
# Documentation: https://matplotlib.org/stable/contents.html

# Read the CSV file into a DataFrame
df = pd.read_csv('csvExamples/courses.csv')
# Documentation: https://pandas.pydata.org/docs/reference/api/pandas.read_csv.html

# Display the first few rows of the DataFrame to inspect the data
df.head()  
# Documentation: https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.head.html

# Set the x-axis label
plt.xlabel('Programming Language')  
# Documentation: https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.xlabel.html

# Set the y-axis label
plt.ylabel('Days to Learn')  
# Documentation: https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.ylabel.html

# Rotate x-axis labels for better visibility 
plt.xticks(rotation=45)
# Documentation: https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.xticks.html

# Set the title of the plot
plt.title('Days to Learn Programming Languages')  
# Documentation: https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.title.html

# Create a bar chart with specified x and y data, and set the bar color to blue
myplt = plt.bar(df['Programming Language'],
                df['Learning Days'],
                color='blue')  
# Documentation: https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.bar.html

# Display the plot
plt.show()  
# Documentation: https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.show.html



# ---- Explanations ----

# 1. Import Libraries:
#    - pandas is imported for data manipulation and analysis.
#    - matplotlib.pyplot is imported for creating visualizations.

# 2. Read CSV File:
#    - The pd.read_csv() method reads the CSV file into a pandas DataFrame.

# 3. Inspect Data:
#    - The head() method displays the first few rows of the DataFrame to inspect the data.

# 4. Set Labels and Title:
#    - The xlabel(), ylabel(), and title() methods set labels and title for the plot.

# 5. Create Bar Chart:
#    - The plt.bar() method creates a bar chart with specified x and y data, and sets the bar color to blue.
#    - The resulting object (myplt) can be used to further customize the plot.

# 6. Display the Plot:
#    - The plt.show() method displays the created plot.
