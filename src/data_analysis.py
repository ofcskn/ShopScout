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

# Main function for command-line interface
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze and Predict Shopping Data")
    parser.add_argument('--analyze', action='store_true', help="Run data analysis")

    args = parser.parse_args()

    if args.analyze:
        analyze_data()
    
 