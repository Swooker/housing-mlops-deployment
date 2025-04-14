import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import joblib

# Load the dataset
data = pd.read_csv("Housing.csv")

# Select only the needed features
X = data[["area", "bedrooms", "bathrooms"]]
y = data["price"]

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Save the model
joblib.dump(model, "model.pkl")
print("✅ Model trained and saved as model.pkl")
