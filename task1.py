import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error

# Load the dataset (ensure the CSV is in the same directory or provide full path)
df = pd.read_csv("dataset.csv")

# ------------------------------------------------------
# Step 1: Data Preparation
# ------------------------------------------------------
# Drop columns that are irrelevant or cannot be used for training
columns_to_drop = [
    "datetimeEpoch",
    "sunriseEpoch",
    "sunsetEpoch",  # time-related metadata
    "severityScore",
    "healthRiskScore",  # drop target (save separately)
]

# Features (X) and Target (y)
X = df.drop(columns=columns_to_drop)
y = df["healthRiskScore"]

# Split the data into training and testing sets (80/20 split)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ------------------------------------------------------
# Step 2: Model Training using Gradient Boosting Regressor
# ------------------------------------------------------
# Gradient Boosting builds an ensemble of shallow trees in sequence
# to minimize prediction errors using gradient descent

model = GradientBoostingRegressor(
    n_estimators=100,  # number of boosting stages
    learning_rate=0.1,  # controls contribution of each tree
    max_depth=5,  # limits individual tree complexity
    random_state=42,  # for reproducibility
)

# Train the model
model.fit(X_train, y_train)

# ------------------------------------------------------
# Step 3: Model Evaluation
# ------------------------------------------------------
# Predict on the test set
y_pred = model.predict(X_test)

# Evaluation Metrics:
# RMSE (Root Mean Squared Error): lower is better
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

# MAPE (Mean Absolute Percentage Error): lower is better
mape = mean_absolute_percentage_error(y_test, y_pred)

# Display the results
print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
print(f"Mean Absolute Percentage Error (MAPE): {mape:.4f}")
