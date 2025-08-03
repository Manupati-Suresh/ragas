
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import pickle

df = pd.read_csv("diabetes.csv")

# Replace 0s with median in important columns
cols = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
for col in cols:
    df[col] = df[col].replace(0, df[col].median())

X = df.drop("Outcome", axis=1)
y = df["Outcome"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

model = LogisticRegression(solver="liblinear")
model.fit(X_train, y_train)

pickle.dump(scaler, open("scaler.pkl", "wb"))
pickle.dump(model, open("logistic_model.pkl", "wb"))
