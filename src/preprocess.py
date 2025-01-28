# preprocess.py
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

def preprocess_data(input_file, output_file):
    # Load data
    data = pd.read_csv(input_file)

    # Fill missing values
    data.fillna(data.median(), inplace=True)

    # Encode categorical features
    label_enc = LabelEncoder()
    data['CategoryColumn'] = label_enc.fit_transform(data['CategoryColumn'])

    # Scale numerical features
    scaler = StandardScaler()
    numerical_features = ['Feature1', 'Feature2', 'Feature3']
    data[numerical_features] = scaler.fit_transform(data[numerical_features])

    # Save processed data
    data.to_csv(output_file, index=False)
    print(f"Data processed and saved to {output_file}")

if __name__ == "__main__":
    preprocess_data("../data/customer_data.csv", "../data/processed_data.csv")
