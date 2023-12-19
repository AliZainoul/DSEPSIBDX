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

plt.figure(figsize=(12, 4))  
# Documentation: https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.figure.html

# Set the main title for the entire plot
plt.suptitle('Comparison of Programming Languages')  
# Documentation: https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.suptitle.html

# Create Subplot 1: Bar chart for "Learning Days"
# Subplot with 1 row, 2 columns, and this is the first subplot
plt.subplot(1, 2, 1)  
# Documentation: https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.subplot.html
plt.bar(df['Programming Language'],
        df['Learning Days'],
        color='blue')  # Create a bar chart with blue bars
# Documentation: https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.bar.html
plt.title('Average Days to Learn')  # Set the title for this subplot
# Documentation: https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.title.html
plt.xlabel('Programming Language')  # Set the x-axis label
# Documentation: https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.xlabel.html
plt.ylabel('Days to Learn')  # Set the y-axis label
# Documentation: https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.ylabel.html
plt.xticks(rotation=45, ha='right')  # Rotate x-axis labels for better visibility
# Documentation: https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.xticks.html


# Create Subplot 2: Bar chart for "Number of Downloads"
# Subplot with 1 row, 2 columns, and this is the second subplot
plt.subplot(1, 2, 2)  
# Documentation: https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.subplot.html
plt.bar(df['Programming Language'],
        df['Number of Downloads'],
        color='orange')  # Create a bar chart with orange bars
# Documentation: https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.bar.html
plt.title('Number of Downloads')  # Set the title for this subplot
# Documentation: https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.title.html
plt.xlabel('Programming Language')  # Set the x-axis label
# Documentation: https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.xlabel.html
plt.ylabel('Number of Downloads')  # Set the y-axis label
# Documentation: https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.ylabel.html
plt.xticks(rotation=45, ha='right')  # Rotate x-axis labels for better visibility
# Documentation: https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.xticks.html

# Automatically adjust subplot parameters for better layout
plt.tight_layout()  
# Documentation: https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.tight_layout.html

# Show the entire plot with subplots
plt.show()  
# Documentation: https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.show.html
