import argparse
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error

def main(data_path, result_save_path, model_save_path):
    # Load your dataset
    df = pd.read_csv(data_path)

    # Extract features and target
    X = df[["6"]]
    y = df["target"]

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Model selection and training
    model = DecisionTreeRegressor(random_state=42)
    model.fit(X_train, y_train)

    # Save the trained model
    joblib.dump(model, model_save_path)

    # Make predictions
    y_pred = model.predict(X_test)

    # Evaluation metrics
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    r2 = r2_score(y_test, y_pred)

    # Display metrics
    print(f"Root mean squared Error: {rmse}")

    # Visualize predictions vs actual values
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], linestyle='--', color='red', linewidth=2)
    plt.xlabel("Actual Values")
    plt.ylabel("Predicted Values")
    plt.title("Actual vs Predicted Values for Regression")
    
    # Save the plot
    plt.savefig(result_save_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Decision Tree Regression algo for anonymized features")
    parser.add_argument("--data_path", type=str, required=True, help="Path to the CSV file containing the dataset")
    parser.add_argument("--result_save_path", type=str, default="result.png", help="Path to save the plot (including filename)")
    parser.add_argument("--model_save_path", type=str, required=True, help="Path to save the trained model")

    args = parser.parse_args()

    main(args.data_path, args.result_save_path, args.model_save_path)
