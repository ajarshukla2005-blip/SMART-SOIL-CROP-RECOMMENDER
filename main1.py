import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

def main():
    print("--- Loading Soil & Environment Data ---")
    # In a real scenario, this is where you load your dataset
    # e.g., df = pd.read_csv('crop_recommendation.csv')
    # For demonstration, I'm simulating the dataframe structure
    data = {
        'Nitrogen': np.random.randint(0, 140, 1000),
        'Phosphorus': np.random.randint(5, 145, 1000),
        'Potassium': np.random.randint(5, 205, 1000),
        'Temperature': np.random.uniform(10.0, 40.0, 1000),
        'Humidity': np.random.uniform(20.0, 95.0, 1000),
        'Target_Crop': np.random.choice(['Rice', 'Maize', 'Chickpea', 'Kidneybeans', 'Pigeonpeas'], 1000)
    }
    df = pd.DataFrame(data)
    
    # 1. Exploratory Data Analysis (EDA)
    print("\nFirst 5 rows of our dataset:")
    print(df.head())
    
    # 2. Data Preprocessing
    # Splitting features (X) and target labels (y)
    X = df.drop('Target_Crop', axis=1)
    y = df['Target_Crop']
    
    # Splitting into training and testing sets (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 3. Model Building
    # I chose Random Forest because it handles non-linear relationships well and is resistant to overfitting
    print("\n--- Training the Random Forest Model ---")
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    
    # 4. Predictions & Evaluation
    y_pred = rf_model.predict(X_test)
    
    print("\n--- Model Evaluation ---")
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model Accuracy: {accuracy * 100:.2f}%")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # 5. Feature Importance (Understanding what drives the decision)
    importances = rf_model.feature_importances_
    features = X.columns
    
    # Plotting Feature Importance
    plt.figure(figsize=(8, 5))
    sns.barplot(x=importances, y=features, palette="viridis")
    plt.title("What matters most for crop selection?")
    plt.xlabel("Importance Score")
    plt.ylabel("Environmental/Soil Factors")
    plt.tight_layout()
    plt.savefig('feature_importance.png')
    print("\nSaved feature importance plot as 'feature_importance.png'")

if __name__ == "__main__":
    main()
