import pandas as pd
import statsmodels.api as sm
from itertools import combinations

# Load the data from the CSV file
file_path = 'spotify.csv'
data = pd.read_csv(file_path, encoding='latin1')

# Get unique genres and define the variables
genres = data['track_genre'].unique()
numeric_columns = data.select_dtypes(include=['number']).columns
variables = [col for col in numeric_columns if col != 'popularity']

# Initialize a dictionary to store the regression formulas
genre_regression_formulas = {}

for genre in genres:
    # Get the relevant data for the genre
    genre_data = data[data['track_genre'] == genre]

    # Prepare the feature matrix X and the target vector y
    X = genre_data[variables]
    y = genre_data['popularity']

    # Fit the base regression model without interaction terms
    X_base = sm.add_constant(X)
    model_base = sm.OLS(y, X_base).fit()
    adjusted_r_squared_base = model_base.rsquared_adj

    # Add interaction terms
    X_interactions = X.copy()
    for (var1, var2) in combinations(variables, 2):
        interaction_term = genre_data[var1] * genre_data[var2]
        interaction_term_name = f"{var1}:{var2}"
        X_interactions[interaction_term_name] = interaction_term

    X_interactions = sm.add_constant(X_interactions)
    model_interactions = sm.OLS(y, X_interactions).fit()
    adjusted_r_squared_interactions = model_interactions.rsquared_adj

    # Decide whether to include interaction terms based on adjusted R-squared
    if adjusted_r_squared_interactions > adjusted_r_squared_base:
        significant_vars = [var for var, pval in zip(X_interactions.columns, model_interactions.pvalues) if pval < 0.05]
        model = model_interactions
    else:
        significant_vars = [var for var, pval in zip(X_base.columns, model_base.pvalues) if pval < 0.05]
        model = model_base

    # Print out model parameters to debug the constant term issue
    print(f"Model parameters for genre '{genre}': {model.params}")

    # Create the formula string with only significant variables
    constant_term_name = model.params.index[0]  # Get the name of the constant term
    formula = f"popularity = {model.params[constant_term_name]:.4f}"
    for variable in significant_vars:
        if variable != constant_term_name:
            formula += f" + {model.params[variable]:.4f} * {variable}"
    formula += " + u"

    # Store the formula in the dictionary
    genre_regression_formulas[genre] = formula

# Print the formulas
for genre, formula in genre_regression_formulas.items():
    print(f"{genre}: {formula}")

# Create a DataFrame from the dictionary
formulas_df = pd.DataFrame(list(genre_regression_formulas.items()), columns=['Genre', 'Regression Formula'])

# Save the DataFrame to a CSV file
output_file = 'genre_regression_formulas_with_interactions.csv'
formulas_df.to_csv(output_file, index=False)

print(f"The regression formulas with interactions have been saved to {output_file}")
