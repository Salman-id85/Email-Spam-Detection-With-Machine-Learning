# Email-Spam-Detection-With-Machine-Learning

# Initialize Project

Set Up Directory Structure: Organize your project into folders such as data/ for datasets, src/ for source code, notebooks/ for Jupyter notebooks, and models/ for storing trained models.
Initialize Git Repository: Use Git to track changes and collaborate with others.
Data Preparation

# Load Dataset: 
Import your dataset into a DataFrame.
Rename Columns: Update column names for clarity and ease of use.
Drop Unnecessary Columns: Retain only the columns relevant to spam detection.
Drop Missing Values: Remove any rows with missing data to ensure data quality.
Map Labels: Convert categorical labels (e.g., 'ham' and 'spam') into binary values (0 and 1).
Split Data: Divide the data into training and testing sets to evaluate the model's performance.
Feature Extraction

# TF-IDF Vectorization: 
Transform the text data into numerical features using the TF-IDF (Term Frequency-Inverse Document Frequency) method. This represents the importance of words in the emails relative to the entire dataset.
Model Training

# Naive Bayes Model:
Train a Multinomial Naive Bayes classifier using the training data. This model is well-suited for text classification tasks due to its simplicity and effectiveness.
Model Evaluation

# Predict:
Use the trained model to generate predictions on the test set.
Evaluate: Assess the model’s performance using metrics such as accuracy, precision, recall, and F1-score. This helps to understand how well the model performs on unseen data.
Model Persistence

# Save Models: 
Store the trained model and vectorizer to disk for future use. This allows you to deploy the model without needing to retrain it and ensures that new data can be transformed using the same vectorizer.
