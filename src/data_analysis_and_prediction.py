import pandas as pd
import argparse
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor

# Load the dataset
file_path = '../data/shopping_trends.csv'
data = pd.read_csv(file_path)


# 1. Analysis Function
def analyze_data():
    # Summary statistics for numerical data
    print("\nSummary Statistics for Age and Purchase Amount (USD):")
    print(data[['Age', 'Purchase Amount (USD)']].describe())
    
    # Average Purchase Amount by Category
    print("\nAverage Purchase Amount by Category:")
    print(data.groupby('Category')['Purchase Amount (USD)'].mean())
    
    # Seasonal Trends
    print("\nAverage Purchase Amount by Season:")
    print(data.groupby('Season')['Purchase Amount (USD)'].mean())
    
    # Distribution of Item Purchased
    print("\nTop 5 Most Purchased Items:")
    print(data['Item Purchased'].value_counts().head(5))
    
    # Review Rating Insights
    print("\nAverage Review Rating by Category:")
    print(data.groupby('Category')['Review Rating'].mean())

# 2. Prediction Function
def predict_purchase_amount(age, item_purchased, category, location, size, color, season, review_rating):
    # Encode categorical variables
    encoder = LabelEncoder()
    for col in ['Item Purchased', 'Category', 'Location', 'Size', 'Color', 'Season']:
        data[col] = encoder.fit_transform(data[col])
    
    # Prepare features and target
    X = data[['Age', 'Item Purchased', 'Category', 'Location', 'Size', 'Color', 'Season', 'Review Rating']]
    y = data['Purchase Amount (USD)']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    train_labels = set(y_train)
    test_labels = set(y_test)

    unseen_labels = test_labels - train_labels
    print("Unseen labels in y_test:", unseen_labels)

    # Filter test set to include only seen labels
    filtered_indices = [i for i, label in enumerate(y_test) if label in train_labels]
    X_test = [X_test[i] for i in filtered_indices]
    y_test = [y_test[i] for i in filtered_indices]

    fallback_label = "Unknown"
    y_test = [label if label in train_labels else fallback_label for label in y_test]

    # Train a Random Forest Regressor
    model = RandomForestRegressor(random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluate the model
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print(f"\nModel Mean Squared Error: {mse}")
    
    # Prepare input for prediction
    input_data = pd.DataFrame({
        'Age': [age],
        'Item Purchased': [encoder.transform([item_purchased])[0]],
        'Category': [encoder.transform([category])[0]],
        'Location': [encoder.transform([location])[0]],
        'Size': [encoder.transform([size])[0]],
        'Color': [encoder.transform([color])[0]],
        'Season': [encoder.transform([season])[0]],
        'Review Rating': [review_rating]
    })
    
    # Predict purchase amount
    predicted_amount = model.predict(input_data)
    print(f"\nPredicted Purchase Amount: {predicted_amount[0]:.2f} USD")

# Main function for command-line interface
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze and Predict Shopping Data")
    parser.add_argument('--analyze', action='store_true', help="Run data analysis")

    args = parser.parse_args()

    if args.analyze:
        analyze_data()
    
 