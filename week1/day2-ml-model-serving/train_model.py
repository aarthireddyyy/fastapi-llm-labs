# train_model.py
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib

def train_and_save(path="model.pkl"):
    data = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, random_state=42)
    model = RandomForestClassifier(n_estimators=50, random_state=42)
    model.fit(X_train, y_train)
    print("Train score:", model.score(X_test, y_test))
    joblib.dump({
        "model": model,
        "target_names": data.target_names.tolist()
    }, path)
    print("Saved model to", path)

if __name__ == "__main__":
    train_and_save()
