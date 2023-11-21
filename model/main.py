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
    """Create EDA visualizations and return them as Markdown-formatted images."""
    # Create pairplot
    pairplot_markdown = markdown.markdown(plt.show(sns.pairplot(data)))

    # Create correlation heatmap
    correlation_matrix = data.corr()
    correlation_heatmap = markdown.markdown(plt.show(sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")))
    plt.close()  # Close plot window

    return pairplot_markdown, correlation_heatmap

if __name__ == "__main__":
    # Generate synthetic data
    house_data = generate_synthetic_data()

    # Visualize EDA and generate Markdown-formatted images
    pairplot_markdown, correlation_heatmap = visualize_eda(house_data)
