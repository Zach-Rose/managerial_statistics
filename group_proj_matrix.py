import pandas as pd
import statsmodels.api as sm
from pprint import pprint, pformat
import pyperclip
import json

# Load the data from the CSV file with error handling for encoding issues
file_path = 'spotify.csv'

# Try to read the file with different encodings
try:
    data = pd.read_csv(file_path, encoding='utf-8')
except UnicodeDecodeError:
    data = pd.read_csv(file_path, encoding='latin1')

# Display the first few rows to verify the data
print(data.head())

# Get unique genres
genres = data['track_genre'].unique()

# Initialize a dictionary to store correlation results and R-squared values for each genre
genre_correlations = {}
genre_r_squared = {}

for genre in genres:
    genre_data = data[data['track_genre'] == genre]
    numeric_columns_genre = genre_data.select_dtypes(include=['number']).columns
    correlation_matrix = genre_data[numeric_columns_genre].corr()
    top_5_correlations = correlation_matrix['popularity'].reindex(
        correlation_matrix['popularity'].abs().sort_values(ascending=False).index).iloc[1:6]
    top_5_correlations_dict = top_5_correlations.to_dict()
    genre_correlations[genre] = top_5_correlations_dict

    # Calculate R-squared values for the top 5 correlated variables
    r_squared_values = {}
    for variable in top_5_correlations_dict.keys():
        X = sm.add_constant(genre_data[[variable]])
        y = genre_data['popularity']
        model = sm.OLS(y, X).fit()
        r_squared_values[variable] = model.rsquared

    genre_r_squared[genre] = r_squared_values

# Pretty print the genre correlations
pprint(genre_correlations)

# Pretty print the genre R-squared values
pprint(genre_r_squared)

# Convert the dictionary to a nicely formatted string
genre_r_squared_str = pformat(genre_r_squared)

# Copy the string to the clipboard
pyperclip.copy(genre_r_squared_str)

print("The genre R-squared values have been copied to the clipboard.")

# Analyze the relationship between correlation coefficients and R-squared values
correlation_vs_r_squared = {}

for genre in genre_correlations:
    correlation_vs_r_squared[genre] = {}
    for variable in genre_correlations[genre]:
        correlation_vs_r_squared[genre][variable] = {
            "correlation": genre_correlations[genre][variable],
            "r_squared": genre_r_squared[genre][variable]
        }

# Pretty print the relationship between correlation coefficients and R-squared values
pprint(correlation_vs_r_squared)

# Convert the analysis to a string and copy to clipboard
correlation_vs_r_squared_str = pformat(correlation_vs_r_squared)
pyperclip.copy(correlation_vs_r_squared_str)

print("The relationship between correlation coefficients and R-squared values has been copied to the clipboard.")
flattened_data = []
for genre, variables in correlation_vs_r_squared.items():
    for variable, stats in variables.items():
        row = {
            "genre": genre,
            "variable": variable,
            "correlation": stats["correlation"],
            "r_squared": stats["r_squared"]
        }
        flattened_data.append(row)

# Create DataFrame
df = pd.DataFrame(flattened_data)

# Save to Excel
output_file = "correlation_matrix_flat.xlsx"
df.to_excel(output_file, index=False)

print(f"The correlation matrix has been saved to {output_file}")
genre_regression_formulas = {}

for genre, variables in correlation_vs_r_squared.items():
    # Get the relevant data for the genre
    genre_data = data[data['track_genre'] == genre]

    # Prepare the feature matrix X and the target vector y
    X = genre_data[list(variables.keys())]
    X = sm.add_constant(X)  # Add a constant term for the intercept
    y = genre_data['popularity']

    # Fit the regression model
    model = sm.OLS(y, X).fit()

    # Filter out variables that are not statistically significant
    significant_variables = {var: coef for var, coef, pval in zip(X.columns[1:], model.params[1:], model.pvalues[1:]) if pval < 0.05}

    # Create the formula string with only significant variables
    formula = f"popularity = {model.params[0]:.4f}"
    for variable, coefficient in significant_variables.items():
        formula += f" + {coefficient:.4f} * {variable}"
    formula += " + u"

    # Store the formula in the dictionary
    genre_regression_formulas[genre] = formula

# Print the formulas
for genre, formula in genre_regression_formulas.items():
    print(f"{genre}: {formula}")

# Create a DataFrame from the dictionary
formulas_df = pd.DataFrame(list(genre_regression_formulas.items()), columns=['Genre', 'Regression Formula'])

# Save the DataFrame to a CSV file
output_file = 'genre_regression_formulas.csv'
formulas_df.to_csv(output_file, index=False)

print(f"The regression formulas have been saved to {output_file}")