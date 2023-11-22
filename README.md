# GHA_MLOps_SelfHostedRunner

# House Price Prediction Analysis Workflow

This GitHub Actions workflow automates the process of generating a house price prediction report and updating the report file in the repository. It utilizes various actions to handle different aspects of the task.

## Trigger Events

The workflow is triggered by the following events:

- Pull requests
- Pushes to any branch
- Manual workflow dispatch

## Jobs

The workflow consists of a single job named `build`.

### build Job

The `build` job runs on the `ubuntu-agastya` runner and performs the following steps:

1. **Checkout Code:** Checks out the code from the repository using the `actions/checkout@v2` action.

2. **Set Up Python:** Sets up the Python environment using the `actions/setup-python@v2` action with Python version `3.11`.

3. **Create Virtual Environment:** Creates a virtual environment named `myenv` using the `python -m venv myenv` command.

4. **Activate Virtual Environment:** Activates the virtual environment using the `source myenv/bin/activate` command.

5. **Install Dependencies:** Installs the required Python dependencies using the following commands:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

6. **Change Permissions of main.py:** Modifies the permissions of the `model/main.py` file to make it executable using the `chmod +x model/main.py` command.

7. **Run Python Script:** Executes the Python script `model/main.py`. This script is responsible for generating the house price prediction report.

8. **Display File Path and Report Contents:** Displays the file path of the generated report (`report.md`) and its contents using the `echo` command.

9. **Set Git Config:** Sets the Git configuration for the user to `actions@github.com` and the name to `GitHub Actions`.

10. **Create or Checkout Branch:** Creates a new branch named `experiment-1` or checks out an existing branch with the same name using the `git checkout experiment-1` command.

11. **Git Add, Commit, and Push:** Adds the updated report file (`report.md`) to the staging area, commits the changes with the message `Update report.md [skip ci]`, and pushes the changes to the `origin` remote repository on the `experiment-1` branch using the following commands:

```bash
git add report.md
git commit -m "Update report.md [skip ci]"
git push origin experiment-1
```

12. **Commit report.md:** Commits the changes made to the `report.md` file with the message `Update report.md [skip ci]` using the following commands:

```bash
git add report.md
git commit -m "Update report.md [skip ci]"
git push
```
This workflow effectively automates the process of generating and updating the house price prediction report, ensuring consistency and reproducibility.

# Model

# House Price Prediction Analysis and Report Generation

This Python script simulates house price data, evaluates different regression models, generates exploratory data analysis (EDA) visualizations, and creates a comprehensive Markdown report.

## Data Generation

The `generate_synthetic_data` function creates a realistic dataset for house price prediction:

- It randomly generates values for features like area, bedrooms, bathrooms, location, lot size, and year built.
- Calculates the house price based on the features and incorporates random noise.
- Stores the generated data in a Pandas DataFrame.
- Encodes categorical columns using one-hot encoding for compatibility with machine learning algorithms.

## Model Evaluation

The `evaluate_models` function assesses the performance of different regression models:

- It separates features (X) from the target variable (y) in the provided dataset.
- Standardizes numerical features using StandardScaler to ensure consistent scales.
- Applies Principal Component Analysis (PCA) to reduce dimensionality and improve computational efficiency.
- Defines a set of regression models to evaluate: Linear Regression, Random Forest, and Gradient Boosting.
- Employs cross-validation to evaluate the models' predictive accuracy using appropriate features (X_scaled for Linear Regression, X_pca for others).
- Calculates the mean squared error (MSE) as the performance metric.
- Returns a dictionary containing the MSE values for each evaluated model.

## Exploratory Data Analysis (EDA)

The `visualize_eda` function generates informative EDA visualizations:

- It creates a pairplot using Seaborn to visualize relationships between features.
- Generates a correlation heatmap using Seaborn to visualize feature correlations.
- Encodes the visualizations into Base64 format for embedding in Markdown reports.

## Report Generation

The main code section combines the data generation, model evaluation, and EDA visualization steps to create a comprehensive report:

- Generates synthetic house price data using `generate_synthetic_data`.
- Creates Markdown-formatted image representations for EDA visualizations using `visualize_eda`.
- Opens the `report.md` file for writing.
- Writes the EDA report header and embeds the pairplot visualization using its Markdown representation.
- Writes the correlation heatmap section and embeds the heatmap visualization using its Markdown representation.
- Closes the `report.md` file after writing the report content.
- Prints the file path of the generated report.md file.

This script effectively automates the process of generating house price data, evaluating regression models, creating EDA visualizations, and compiling a comprehensive Markdown report.



