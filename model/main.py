import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import markdown
import io
import base64
import os

def generate_synthetic_data():
    """Generate synthetic data for house price prediction."""
    np.random.seed(42)

    # Generating multiple features
    area = np.random.randint(800, 5000, 100)
    bedrooms = np.random.randint(1, 6, 100)
    bathrooms = np.random.randint(1, 4, 100)
    location = np.random.choice(['Downtown', 'Suburb', 'Rural'], 100)
    lot_size = np.random.randint(3000, 15000, 100)
    year_built = np.random.randint(1970, 2022, 100)

    price = 100 * area + 20000 * bedrooms + 15000 * bathrooms + 3000 * lot_size + (2022 - year_built) * 500 \
            + np.random.randn(100) * 50000

    data = pd.DataFrame({
        'Area': area,
        'Bedrooms': bedrooms,
        'Bathrooms': bathrooms,
        'Location': location,
        'LotSize': lot_size,
        'YearBuilt': year_built,
        'Price': price
    })
    # Identify and handle categorical columns (for example 'Location' is a categorical column)
    categorical_columns = ['Location']

    # Perform one-hot encoding for categorical columns
    data = pd.get_dummies(data, columns=categorical_columns)

    return data

def evaluate_models(data):
    """Evaluate different regression models using cross-validation and return model metrics."""
    X = data.drop('Price', axis=1)
    y = data['Price']

    # Standardize numerical features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Apply PCA to reduce dimensionality
    pca = PCA(n_components=3)
    X_pca = pca.fit_transform(X_scaled)

    # Define and evaluate different regression models
    models = {
        'Linear Regression': LinearRegression(),
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42)
    }

    model_metrics = {}
    for model_name, model in models.items():
        if model_name == 'Linear Regression':
            scores = cross_val_score(model, X_scaled, y, cv=5, scoring='neg_mean_squared_error')
        else:
            scores = cross_val_score(model, X_pca, y, cv=5, scoring='neg_mean_squared_error')
        mse_mean = -scores.mean()
        model_metrics[model_name] = mse_mean

    return model_metrics


def visualize_eda(data):
    # Create pairplot
    pairplot = sns.pairplot(data)
    pairplot_buffer = io.BytesIO()
    pairplot.savefig(pairplot_buffer, format='png')
    pairplot_encoded = base64.b64encode(pairplot_buffer.getvalue()).decode('utf-8')
    pairplot_markdown = f"![Pairplot](data:image/png;base64,{pairplot_encoded})"

    # Create correlation heatmap
    correlation_matrix = data.corr()
    heatmap = sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
    heatmap_buffer = io.BytesIO()
    heatmap.figure.savefig(heatmap_buffer, format='png')
    heatmap_encoded = base64.b64encode(heatmap_buffer.getvalue()).decode('utf-8')
    correlation_heatmap = f"![Correlation Heatmap](data:image/png;base64,{heatmap_encoded})"

    plt.close('all')  # Close all plots

    return pairplot_markdown, correlation_heatmap

if __name__ == "__main__":
    house_data = generate_synthetic_data()

    pairplot_markdown, correlation_heatmap = visualize_eda(house_data)

    # Save the pairplot and correlation heatmap as image files
    pairplot = sns.pairplot(house_data)
    pairplot.savefig('pairplot.png')
    plt.close(pairplot.fig)

    correlation_matrix = house_data.corr()
    heatmap = sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
    heatmap.figure.savefig('correlation_heatmap.png')
    plt.close(heatmap.figure)

    # Write Markdown content to report.md
    with open('report.md', 'w') as file:
        file.write("# EDA Report\n\n")
        file.write("## Pairplot Visualization\n")
        file.write("![Pairplot](pairplot.png)\n\n")
        file.write("## Correlation Heatmap\n")
        file.write("![Correlation Heatmap](correlation_heatmap.png)\n\n")


        # Add a section for model metrics
        file.write("\n\n## Model Metrics\n")
        file.write("| Model | Mean Squared Error |\n")
        file.write("|-------|--------------------|\n")

        model_metrics = evaluate_models(house_data)
        for model, mse in model_metrics.items():
            file.write(f"| {model} | {mse:.2f} |\n")

    # Print the file path where report.md is located
    file_path = 'report.md'
    print(f"File created at path: {os.path.abspath(file_path)}")

    # Read the contents of report.md and print it
    with open('report.md', 'r') as file:
        report_contents = file.read()
        print(report_contents)
    print(f"File created at path: {os.path.abspath(file_path)}")

    # Read the contents of report.md and print it
    with open('report.md', 'r') as file:
        report_contents = file.read()
        print(report_contents)
