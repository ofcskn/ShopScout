import argparse
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error

# Error handling for the ValueError
def handle_value_error(error):
    print(f"Error: {error}")
    print("Possible cause: The dataset may contain previously unseen labels in the target variable.")
    exit(1)


# Parse arguments from terminal
def parse_args():
    parser = argparse.ArgumentParser(description='Predict Purchase Amount')
    parser.add_argument('--item', type=str, required=True, help='Item Purchased')
    parser.add_argument('--category', type=str, required=True, help='Category')
    parser.add_argument('--location', type=str, required=True, help='Location')
    parser.add_argument('--size', type=str, required=True, help='Size')
    parser.add_argument('--color', type=str, required=True, help='Color')
    return parser.parse_args()

# List unique values for the input columns
def list_unique_values(data, invalid_args):
    if 'item' in invalid_args:
        print(f"Available unique values for 'Item Purchased': {data['Item Purchased'].unique()}")
    if 'category' in invalid_args:
        print(f"Available unique values for 'Category': {data['Category'].unique()}")
    if 'location' in invalid_args:
        print(f"Available unique values for 'Location': {data['Location'].unique()}")
    if 'size' in invalid_args:
        print(f"Available unique values for 'Size': {data['Size'].unique()}")
    if 'color' in invalid_args:
        print(f"Available unique values for 'Color': {data['Color'].unique()}")


# Main function to predict Purchase Amount
def main():
    try:
        # Load dataset and skip the first row which contains headers
        columns = ["Customer ID", "Age", "Gender", "Item Purchased", "Category", "Purchase Amount (USD)",
                   "Location", "Size", "Color", "Season", "Review Rating", "Subscription Status", 
                   "Shipping Type", "Discount Applied", "Promo Code Used", "Previous Purchases", 
                   "Payment Method", "Frequency of Purchases"]
        
        data = pd.read_csv("../data/shopping_trends.csv", names=columns, header=0)  # header=0 ensures first row is used as headers
        # Parse args
        args = parse_args()
        # List of invalid arguments
        invalid_args = []

        # Check if user inputs are valid
        if args.item not in data['Item Purchased'].unique():
            print(f"Error: '{args.item}' is not a valid Item Purchased.")
            invalid_args.append('item')
        if args.category not in data['Category'].unique():
            print(f"Error: '{args.category}' is not a valid Category.")
            invalid_args.append('category')
        if args.location not in data['Location'].unique():
            print(f"Error: '{args.location}' is not a valid Location.")
            invalid_args.append('location')
        if args.size not in data['Size'].unique():
            print(f"Error: '{args.size}' is not a valid Size.")
            invalid_args.append('size')
        if args.color not in data['Color'].unique():
            print(f"Error: '{args.color}' is not a valid Color.")
            invalid_args.append('color')

        # If any invalid argument, list available values
        if invalid_args:
            list_unique_values(data, invalid_args)
            exit(1)

        # Select relevant features for training
        features = data[["Item Purchased", "Category", "Location", "Size", "Color"]]
        target = data["Purchase Amount (USD)"]
        
        label_encoders = {}
        for column in features.columns:
            if features[column].dtype == object:  # Only encode categorical columns
                le = LabelEncoder()
                features[column] = le.fit_transform(features[column].astype(str))
                label_encoders[column] = le
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
        
        # Train RandomForestRegressor
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        # Evaluate the model
        y_pred = model.predict(X_test)
        error = mean_absolute_error(y_test, y_pred)
        print(f"Model Mean Absolute Error: {error:.2f}")
        
        # Get input from args
        input_data = [[args.item, args.category, args.location, args.size, args.color]]
        
        # Label encode the input data
        for idx, column in enumerate(["Item Purchased", "Category", "Location", "Size", "Color"]):
            input_data[0][idx] = label_encoders[column].transform([input_data[0][idx]])[0]
        
        # Predict purchase amount
        prediction = model.predict(input_data)
        print(f"Predicted Purchase Amount (USD): {prediction[0]:.2f}")
    
    except ValueError as e:
        handle_value_error(e)

if __name__ == "__main__":
    main()


# python predict_purchase.py --item "Blouse" --location "California" --size "L" --color "Gray" --category "Clothing"