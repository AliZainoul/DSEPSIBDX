# Import necessary libraries

# For data manipulation and analysis
import pandas as pd
# Documentation: https://pandas.pydata.org/docs/

# For creating visualizations
import matplotlib.pyplot as plt  
# Documentation: https://matplotlib.org/stable/contents.html

# Read the CSV file into a DataFrame
df = pd.read_csv('csvExamples/randomStudentData.csv')
# Documentation: https://pandas.pydata.org/docs/reference/api/pandas.read_csv.html

# Calculate the number of students
num_students = len(df['StudentID'])  
# Documentation: https://docs.python.org/3/library/functions.html#len

# Calculate the figure size based on the number of students
minW = 6
minH = 8
# Minimum width of minW inches
fig_width = max(minW, num_students) 
# Minimum height of minH inches
fig_height = max(minH, num_students)  

# Create subplots with 1 row and 1 column
# Create a figure and a set of subplots 
fig, ax1 = plt.subplots(figsize=(fig_width, fig_height * 0.75)) 
# Documentation: https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.subplots.html

# Set the main title for the entire plot
plt.title('Comparison of Student Data')  
# Documentation: https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.title.html

# Adjust the width as needed
bar_width = 0.4  

# Set x-axis ticks and labels
bar_positions = range(num_students)
# Set the x-axis tick positions
ax1.set_xticks(bar_positions)  
# Documentation: https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.set_xticks.html
# Rotate x-axis labels for better visibility
ax1.set_xticklabels(df['StudentID'], rotation=45, ha='right')
# Documentation: https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.set_xticklabels.html

# Bar chart for "Marks"
# Adjust alpha to control transparency
ax1.bar(bar_positions,
        df['Marks'],
        width=bar_width,
        color='blue', label='Marks', alpha=0.7)  
# Documentation: https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.bar.html

# Set the x-axis label
ax1.set_xlabel('Student ID')
# Documentation: https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.set_xlabel.html

# Set the y-axis label and color
ax1.set_ylabel('Marks', color='blue')  
# Documentation: https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.set_ylabel.html

# Set properties for y-axis ticks
ax1.tick_params('y', colors='blue')  
# Documentation: https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.tick_params.html

# Create a second Y-axis for "IQ"
# Create a twin Axes sharing the xaxis
ax2 = ax1.twinx()  
# Documentation: https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.twinx.html
# Adjust alpha to control transparency
ax2.bar(bar_positions,
        df['IQ'],
        width=bar_width,
        color='green', label='IQ', alpha=0.7) 
# Documentation: https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.bar.html 
# Set the y-axis label and color
ax2.set_ylabel('IQ', color='green')  
# Documentation: https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.set_ylabel.html
# Set properties for y-axis ticks
ax2.tick_params('y', colors='green') 
# Documentation: https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.tick_params.html 

# Show legends
# Display legend for ax1
ax1.legend(loc='upper left')
# Documentation: https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.legend.html
# Display legend for ax2 
ax2.legend(loc='upper right')  
# Documentation: https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.legend.html

# Show the entire plot with subplots
plt.show()   
# Documentation: https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.show.html
