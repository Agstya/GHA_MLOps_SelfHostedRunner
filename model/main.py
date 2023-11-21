import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import make_scorer, mean_squared_error

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

    return data


def evaluate_models(data):
    """Evaluate different regression models using cross-validation."""
    # Splitting features and target variable
    X = data.drop('Price', axis=1)
    y = data['Price']

    # Feature scaling using StandardScaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Apply PCA for dimensionality reduction
    pca = PCA(n_components=3)
    X_pca = pca.fit_transform(X_scaled)

    # Define models
    models = {
        'Linear Regression': LinearRegression(),
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42)
    }

    # Evaluate models using cross-validation
    scoring = make_scorer(mean_squared_error, greater_is_better=False)
    results = {}
    for model_name, model in models.items():
        if model_name == 'Linear Regression':
            scores = cross_val_score(model, X_scaled, y, cv=5, scoring=scoring)
        else:
            scores = cross_val_score(model, X_pca, y, cv=5, scoring=scoring)
        results[model_name] = scores

    # Display mean MSE values for each model
    for model_name, scores in results.items():
        print(f"Model: {model_name}")
        print(f"Mean MSE: {-scores.mean():.2f} (+/- {scores.std():.2f})")
        print("=" * 40)


def visualize_eda(data):
    """Create EDA visualizations and save them as an image."""
    # Create pairplot for relationships between variables
    sns.pairplot(data)
    plt.title('Pairplot - Relationships between Variables')
    plt.savefig('pairplot.png')  # Save pairplot as an image
    plt.close()

    # Correlation matrix heatmap
    correlation_matrix = data.corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Correlation Matrix')
    plt.savefig('correlation_heatmap.png')  # Save heatmap as an image
    plt.close()

    # Summary statistics table visualization (optional)
    summary_table = data.describe()
    summary_table.to_csv('summary_statistics.csv')  # Save summary statistics as a CSV file


if __name__ == "__main__":
    # Generate synthetic data
    house_data = generate_synthetic_data()

    # Visualize EDA
    visualize_eda(house_data)

    # Evaluate models
    evaluate_models(house_data)
