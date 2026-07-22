from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
import joblib

iris = load_iris()

model = RandomForestClassifier(n_estimators=100, random_state=42)

model.fit(iris.data, iris.target)

joblib.dump(
    {
        "model": model,
        "feature_names": iris.feature_names,
        "target_names": iris.target_names,
    },
    "iris_classifier.pkl",
)

print("Model saved successfully!")
