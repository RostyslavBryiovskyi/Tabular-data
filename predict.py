import argparse
import joblib
import pandas as pd
import numpy as np

def main(data_path, model_path, predictions_save_path):

    # Load the trained model
    model = joblib.load(model_path)

    # Load the test set
    df_test = pd.read_csv(data_path)

    # Extract features for the test set
    X_test_real = df_test[["6"]]

    # Make predictions for the test set
    y_pred_real = model.predict(X_test_real)

    # Save predictions to a file
    df_predictions = pd.DataFrame({'Predictions': y_pred_real})
    df_predictions.to_csv(predictions_save_path, index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Code for model prediction.")
    parser.add_argument("--data_path", type=str, required=True, help="Path to the CSV file containing the dataset")
    parser.add_argument("--model_path", type=str, required=True, help="Path to saved model")
    parser.add_argument("--predictions_save_path", type=str, default="test_predictions.csv", help="Path to save the predictions for the test set")

    args = parser.parse_args()

    main(args.data_path, args.model_path, args.predictions_save_path)
