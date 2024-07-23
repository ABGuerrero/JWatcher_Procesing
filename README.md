# Data Analysis and Visualization Pipeline
This repository provides a comprehensive pipeline for processing, analyzing, and visualizing trial data from .dat and .csv files. The pipeline includes normalizing date formats, calculating error scores, combining CSV files, fitting linear mixed models, and creating various visualizations.

## Table of Contents

- [Requirements](#requirements)
- [File and Folder Structure](#file-and-folder-structure)
  - [Data Processing Scripts](#data-processing-scripts)
  - [Analysis and Visualization Scripts](#analysis-and-visualization-scripts)
- [Usage](#usage)
  - [Processing `.dat` Files](#processing-dat-files)
  - [Classify Errors](#classify-errors)
  - [Plot Error Data](#plot-error-data)
  - [Combining CSV Files](#combining-csv-files)
  - [Fitting the Linear Mixed Model](#fitting-the-linear-mixed-model)
  - [Summarizing and Plotting Data](#summarizing-and-plotting-data)

## Requirements
- Python 3.9 or later
- pandas
- seaborn
- matplotlib
- statsmodels
- numpy

You can install the required packages using pip:
```python
pip install pandas seaborn matplotlib statsmodels numpy
```
## File and Folder Structure
### Data Processing Scripts

- process_files_in_folder(folder_path, RewZone): Processes .dat files in the specified folder, normalizes dates, and calculates error scores.
- classify_errors(csv_file): Classifies errors and adds them to the results CSV.
- plot_mean_absolute_error_with_sem(csv_file): Plots mean absolute error with SEM.
- plot_error_frequency(csv_file): Plots the frequency of different error types.
- plot_stack_bar_error_frequency(csv_file): Plots a stacked bar chart of error frequencies.
- plot_error_percentage(csv_file): Plots the percentage of each error type.
  
### Analysis and Visualization Scripts
- combine_csv_files.py: Combines CSV files from subfolders into a single DataFrame.
- fit_mixed_model.py: Fits a linear mixed model to the combined data.
- summarize_and_plot.py: Summarizes error counts and creates visualizations.

## Usage
### Processing .dat Files
Read and process .dat files: Use the function process_files_in_folder to handle .dat files, normalize dates, and compute error scores.

```python
process_files_in_folder('X:/MATT_SCORING', 5)
```
### Classify errors 
After processing, use classify_errors to classify errors in the resulting CSV.
```python
classify_errors('X:/MATT_SCORING/MEA_Results_RMB4.csv')
```
### Plot error data 
Generate various plots such as mean absolute error with SEM, error frequency, stacked bar charts, and error percentages.
```python
plot_mean_absolute_error_with_sem('X:/MATT_SCORING/MEA_Results_RMB4.csv')
plot_error_frequency('X:/MATT_SCORING/MEA_Results_RMB4.csv')
plot_error_percentage('X:/MATT_SCORING/MEA_Results_RMB4.csv')
```
### Combining CSV Files
Use combine_csv_files.py to combine all CSV files from subfolders into a single DataFrame:

```python
import os
import pandas as pd

def combine_csv_files(root_folder):
    all_data = []
    
    for subdir, dirs, files in os.walk(root_folder):
        for file in files:
            if file.endswith(".csv"):
                file_path = os.path.join(subdir, file)
                df = pd.read_csv(file_path)
                animal_id = os.path.basename(subdir).split('_')[0]
                df['Animal_ID'] = animal_id
                all_data.append(df)
    
    if all_data:
        combined_df = pd.concat(all_data, ignore_index=True)
        combined_filename = os.path.join(root_folder, 'combined_data.csv')
        combined_df.to_csv(combined_filename, index=False)
        print(f"Combined data saved to {combined_filename}")
    else:
        print("No CSV files found in the specified folder.")

root_folder = 'X:/MATT_SCORING'  # Replace with the root path to your folder containing all the CSVs
combine_csv_files(root_folder)
```
### Fitting the Linear Mixed Model
Use fit_mixed_model.py to fit a linear mixed model to the combined data:

```python
import pandas as pd
import statsmodels.formula.api as smf

# Load the combined dataset
combined_data = pd.read_csv("X:/MATT_SCORING/combined_data.csv")

# Convert Animal_ID, Age, and Group to categorical data type
combined_data['Animal_ID'] = combined_data['Animal_ID'].astype('category')
combined_data['Age'] = combined_data['Age'].astype('category')
combined_data['Group'] = combined_data['Group'].astype('category')

# Define the formula for the linear mixed model
formula = 'Error ~ C(Age) * C(Group)'

# Fit the linear mixed model
model = smf.mixedlm(formula, combined_data, groups=combined_data['Animal_ID'])
result = model.fit()

# Print the summary of the model
print(result.summary())
```

### Summarizing and Plotting Data
Use summarize_and_plot.py to summarize error counts and create visualizations:

```python
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the combined dataset
combined_data = pd.read_csv("X:/MATT_SCORING/combined_data.csv")

# Count the number of errors for each Error_Type within each ID and Age
error_counts = combined_data.groupby(['Animal_ID', 'Age', 'Error_Type', 'Group']).size().reset_index(name='Count')

# Calculate the mean and SEM across groups for each Age
summary_df = error_counts.groupby(['Age', 'Group', 'Error_Type']).agg({'Count': ['mean', 'sem']}).reset_index()
summary_df.columns = ['Age', 'Group', 'Error_Type', 'Mean', 'SEM']

age_categories = {6: '6 months', 9: '9 months'}

# Define functions for age category and order
def age_to_category(age):
    return age_categories.get(age, 'Other')  # Handle missing ages

summary_df['Age_Group'] = summary_df['Age'].apply(age_to_category)
summary_df['Age_Group'] = summary_df['Age_Group'].astype(str)  # Ensure string type

def get_age_group_order(df):
    # Get unique categories in the desired order (adjust order as needed)
    return ['6 months', '9 months', 'Other']

# Create the bar plot with error bars and faceting by Group
age_group_order = get_age_group_order(summary_df)
g = sns.FacetGrid(summary_df, col='Group', hue='Error_Type', col_wrap=2, palette='colorblind')
g.map(sns.barplot, x='Age_Group', y='Mean', ci='sem', order=age_group_order)

# Add labels and title
g.fig.suptitle('Mean Number of Errors by Age, Group, and Error Type', fontsize=12)  # Set suptitle for all subplots
g.fig.subplots_adjust(top=0.88)  # Adjust spacing between title and subplots

# Rotate x-axis labels for better readability if needed
plt.xticks(rotation=45)

plt.show()
```
