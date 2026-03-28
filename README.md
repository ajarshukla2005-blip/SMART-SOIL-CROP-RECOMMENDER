
🌱 Smart-Soil: ML-Powered Crop Recommendation
What does this project do?
Farming is increasingly unpredictable. Often, farmers rely on traditional knowledge or guesswork to decide what to plant, which can lead to poor yields or the overuse of fertilizers (harming the local water supply). Smart-Soil is a fundamental Machine Learning project that takes the guesswork out of planting. By analyzing soil nutrient levels—specifically the delicate balance of Nitrogen, Phosphorus, and Potassium (NPK)—alongside environmental factors like temperature and humidity, it predicts the most optimal crop to plant in that specific soil.

How to Set It Up
You don't need a massive supercomputer to run this. Just a standard Python environment.

Clone the repository: git clone https://github.com/yourusername/SMART-SOIL-CROP-RECOMMENDER.git

Navigate to the directory:
cd smart-soil

Install the required libraries:
Create a virtual environment (optional but recommended) and run:
pip install pandas numpy scikit-learn matplotlib seaborn

Download the Data:
Ensure you have the crop_recommendation.csv dataset in the root folder
How to Use It
Simply run the main script from your terminal:
python main.py

The script will:

Load and clean the dataset.

Train a Random Forest Classifier on the historical environmental data.

Output an accuracy score and a detailed classification report.

Generate a feature_importance.png graph showing whether Nitrogen, Phosphorus, or Temperature was the biggest deciding factor for the model.
