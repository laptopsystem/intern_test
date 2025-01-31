import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Fetching data from API endpoints
quiz_data_url = "https://jsonkeeper.com/b/LLQT"  # Quiz Data
quiz_submission_url = "https://api.jsonserve.com/rJvd7g"  # Quiz Submission Data
historical_data_url = "https://api.jsonserve.com/XgAgFJ"  # Historical Data

def fetch_data(url):
    response = requests.get(url)
    data = response.json()
    return data

# Fetching data
quiz_data = fetch_data(quiz_data_url)
quiz_submission_data = fetch_data(quiz_submission_url)
historical_data = fetch_data(historical_data_url)

# Convert data into DataFrame
quiz_df = pd.DataFrame(quiz_data)
submission_df = pd.DataFrame(quiz_submission_data)
history_df = pd.DataFrame(historical_data)

# Data Preprocessing: Handle missing values and normalize data
quiz_df.fillna(0, inplace=True)
submission_df.fillna(0, inplace=True)
history_df.fillna(0, inplace=True)

# Assume quiz_df contains columns like 'question_id', 'topic', 'difficulty', 'response', 'score'
# Assume history_df contains 'student_id', 'quiz_id', 'score', 'time_spent', 'response_map'
# Assume submission_df contains 'student_id', 'quiz_id', 'selected_option', 'correct_option'

# Feature extraction for analysis and prediction
data = pd.merge(submission_df, history_df, on='student_id', how='inner')
data = pd.merge(data, quiz_df, on='question_id', how='inner')

# Calculate features: Response accuracy, average score, etc.
data['response_accuracy'] = (data['selected_option'] == data['correct_option']).astype(int)
data['average_score'] = data.groupby('student_id')['score'].transform('mean')  # Average score across all quizzes

# Visualizations (For analyzing student performance)
sns.barplot(x='topic', y='score', data=data)
plt.title('Student Performance by Topic')
plt.show()

# Rank Prediction Model
X = data[['average_score', 'response_accuracy', 'time_spent']]
y = data['neet_rank']  # Target: NEET rank prediction

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")

# Visualize the model performance
plt.scatter(y_test, y_pred)
plt.xlabel('True Rank')
plt.ylabel('Predicted Rank')
plt.title('True vs Predicted NEET Rank')
plt.show()

# Generate insights based on data
def generate_insights(data):
    # Weak Areas (Topics where students perform poorly)
    weak_areas = data.groupby('topic')['score'].mean().sort_values(ascending=True).head(5)
    print("Weak Areas (Topics where students perform poorly):")
    print(weak_areas)

    # Improvement Trends (Students who improved in their scores)
    improvement_trends = data.groupby('student_id')['average_score'].mean().sort_values(ascending=False).head(5)
    print("Top 5 Students with Improvement Trends:")
    print(improvement_trends)

generate_insights(data)

# Predict the most likely college based on predicted rank (Bonus)
def predict_college(rank):
    if rank <= 500:
        return "Top Medical Colleges"
    elif rank <= 1000:
        return "Mid-range Medical Colleges"
    else:
        return "Private Medical Colleges"

# Example: Predict college based on a student's rank
example_rank = 450  # Example predicted NEET rank
predicted_college = predict_college(example_rank)
print(f"Predicted College: {predicted_college}")

