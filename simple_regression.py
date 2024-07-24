import pandas as pd
import statsmodels.api as sm

# Load the data from the CSV file
file_path = 'spotify.csv'
data = pd.read_csv(file_path, encoding='latin1')

# Get unique genres
genres = data['track_genre'].unique()

# Get all numeric variables in the dataset (excluding 'popularity' which will be the dependent variable)
numeric_columns = data.select_dtypes(include=['number']).columns
variables = [col for col in numeric_columns if col != 'popularity']

# Initialize a list to store the results
results = []

# Iterate over each genre and variable to perform simple linear regression
for genre in genres:
    for variable in variables:
        genre_data = data[data['track_genre'] == genre]
        X = genre_data[[variable]]
        y = genre_data['popularity']
        X = sm.add_constant(X)

        # Fit the simple linear regression model
        model = sm.OLS(y, X).fit()

        # Extract the coefficient, p-value, and R-squared value
        coefficient = model.params[variable]
        p_value = model.pvalues[variable]
        r_squared = model.rsquared

        # Append the results to the list
        results.append({
            'genre': genre,
            'variable': variable,
            'simple regression coefficient': coefficient,
            'P-Value': p_value,
            'R-squared value': r_squared
        })

# Create a DataFrame from the results
results_df = pd.DataFrame(results)

# Display the DataFrame
print(results_df)

# Save the DataFrame to a CSV file
output_file = 'regression_results.csv'
results_df.to_csv(output_file, index=False)

print(f"The regression results have been saved to {output_file}")
