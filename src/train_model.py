import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

def load_data(path):
    df = pd.read_csv(path)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df = df.dropna()
    return df

def preprocess(df):
    X = df.drop(["Churn", "customerID"], axis=1)
    y = df["Churn"].map({"Yes":1, "No":0})
    X = pd.get_dummies(X, drop_first=True)
    return X, y

def train(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    model = LogisticRegression(max_iter=2000)
    model.fit(X_train, y_train)

    y_prob = model.predict_proba(X_test)[:,1]
    auc = roc_auc_score(y_test, y_prob)
    print("ROC-AUC:", auc)

if __name__ == "__main__":
    # Get project root directory
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # Construct dataset path
    data_path = os.path.join(BASE_DIR, "data", "telco_churn.csv")
    
    df = load_data(data_path)
    X, y = preprocess(df)
    train(X, y)