# src/model.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import pickle
import json
from sklearn.metrics import r2_score

def main():
    df = pd.read_csv("data/rice.csv")
    X = df.drop("yield", axis=1)
    y = df["yield"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Save model and feature order
    with open("rice_model.pkl", "wb") as f:
        pickle.dump(model, f)
    with open("feature_names.json", "w") as f:
        json.dump(list(X.columns), f)

    preds = model.predict(X_test)
    print("R2 on test (quick):", round(r2_score(y_test, preds), 3))
    print("Saved rice_model.pkl and feature_names.json")

if __name__ == "__main__":
    main()
