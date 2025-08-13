




Task-2: Predictive Analysis using Machine Learning

📌 Task Overview

Goal: Create a Machine Learning model that predicts an output based on a dataset.

Example types of prediction:

Regression → Predicting a continuous value (e.g., house price, temperature, sales amount).

Classification → Predicting a category (e.g., spam/ham, pass/fail, disease/no disease).


Deliverable:
A Jupyter Notebook showing:

1. Feature selection (choosing which columns to use)


2. Model training


3. Model evaluation (accuracy, RMSE, etc.)





---

1. Choosing a Dataset

Pick a dataset based on your prediction type:

Regression Examples:

House Price Prediction (Kaggle dataset)

Car Price Prediction

Weather Prediction


Classification Examples:

Titanic Survival Prediction

Diabetes Prediction

Email Spam Classification




---

2. Tools You Can Use

Python Libraries:

pandas → For data handling

scikit-learn → For machine learning models

matplotlib, seaborn → For visualization


Platform:

Google Colab

Kaggle Notebooks

Jupyter Notebook




---

3. Example Machine Learning Workflow

Here’s a simple Classification Example (Titanic Survival Prediction):

# Install necessary libraries
!pip install pandas scikit-learn seaborn matplotlib

# Import Libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load Dataset
url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
df = pd.read_csv(url)

# Select Features
features = ["Pclass", "Sex", "Age", "Fare"]
df["Sex"] = df["Sex"].map({"male": 0, "female": 1})
df = df.dropna(subset=features)
X = df[features]
y = df["Survived"]

# Split Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Predict
predictions = model.predict(X_test)

# Evaluate
print("Accuracy:", accuracy_score(y_test, predictions))
print(classification_report(y_test, predictions))


---

4. Deliverables

1. Notebook (.ipynb) with:

Data loading

Feature selection

Model training

Evaluation metrics



2. Screenshots of results


3. Insights Summary (e.g., “Model accuracy: 82%, females had higher survival rates”)






