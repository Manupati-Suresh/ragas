
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import pickle

# Load iris dataset
iris = load_iris()
X = iris.data
y = iris.target
feature_names = iris.feature_names

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Decision Tree model
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

# Save model
pickle.dump(model, open("decision_tree_model.pkl", "wb"))
