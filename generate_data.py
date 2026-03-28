import pandas as pd
import numpy as np

def generate_crop_data():
    print("Generating synthetic environmental and soil data...")
    
    # Set a random seed so the data is reproducible
    np.random.seed(42)
    
    # Number of data points per crop
    n_samples = 400

    # 1. Rice (Requires heavy water/humidity, high nitrogen)
    rice = pd.DataFrame({
        'Nitrogen': np.random.randint(60, 130, n_samples),
        'Phosphorus': np.random.randint(35, 60, n_samples),
        'Potassium': np.random.randint(35, 45, n_samples),
        'Temperature': np.random.uniform(20.0, 40.0, n_samples),
        'Humidity': np.random.uniform(80.0, 100.0, n_samples),
        'Target_Crop': ['Rice'] * n_samples
    })

    # 2. Chickpea (Tolerates drier conditions, specific phosphorus needs)
    chickpea = pd.DataFrame({
        'Nitrogen': np.random.randint(20, 60, n_samples),
        'Phosphorus': np.random.randint(55, 80, n_samples),
        'Potassium': np.random.randint(75, 85, n_samples),
        'Temperature': np.random.uniform(15.0, 30.0, n_samples),
        'Humidity': np.random.uniform(10.0, 25.0, n_samples),
        'Target_Crop': ['Chickpea'] * n_samples
    })

    # 3. Kidney Beans (Moderate climate, balanced soil nutrients)
    kidneybeans = pd.DataFrame({
        'Nitrogen': np.random.randint(10, 40, n_samples),
        'Phosphorus': np.random.randint(55, 75, n_samples),
        'Potassium': np.random.randint(15, 25, n_samples),
        'Temperature': np.random.uniform(15.0, 25.0, n_samples),
        'Humidity': np.random.uniform(15.0, 25.0, n_samples),
        'Target_Crop': ['Kidneybeans'] * n_samples
    })

    # 4. Maize (Versatile, but prefers moderate-to-high temperature)
    maize = pd.DataFrame({
        'Nitrogen': np.random.randint(60, 100, n_samples),
        'Phosphorus': np.random.randint(35, 60, n_samples),
        'Potassium': np.random.randint(15, 25, n_samples),
        'Temperature': np.random.uniform(18.0, 35.0, n_samples),
        'Humidity': np.random.uniform(50.0, 75.0, n_samples),
        'Target_Crop': ['Maize'] * n_samples
    })

    # Combine all the individual crop dataframes into one master dataset
    final_df = pd.concat([rice, chickpea, kidneybeans, maize], ignore_index=True)

    # Shuffle the dataset so the model doesn't just read all the 'Rice' in order
    final_df = final_df.sample(frac=1, random_state=42).reset_index(drop=True)

    # Save to CSV
    filename = 'crop_recommendation.csv'
    final_df.to_csv(filename, index=False)
    
    print(f"Success! Generated {len(final_df)} rows of data and saved to '{filename}'.")
    print("You can now run 'python main.py' to train your model!")

if __name__ == "__main__":
    generate_crop_data()
