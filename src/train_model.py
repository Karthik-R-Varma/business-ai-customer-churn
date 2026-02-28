import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, recall_score
import matplotlib.pyplot as plt

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
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # -------------------
    # Logistic Regression
    # -------------------
    log_model = LogisticRegression(max_iter=2000)
    log_model.fit(X_train_scaled, y_train)

    y_prob_log = log_model.predict_proba(X_test_scaled)[:, 1]
    y_pred_log = (y_prob_log > 0.4).astype(int)

    log_auc = roc_auc_score(y_test, y_prob_log)
    log_acc = accuracy_score(y_test, y_pred_log)
    log_recall = recall_score(y_test, y_pred_log)

    # -------------------
    # Random Forest
    # -------------------
    rf_model = RandomForestClassifier(
        n_estimators=200,
        random_state=42
    )

    rf_model.fit(X_train, y_train)

    y_prob_rf = rf_model.predict_proba(X_test)[:, 1]
    y_pred_rf = (y_prob_rf > 0.4).astype(int)

    rf_auc = roc_auc_score(y_test, y_prob_rf)
    rf_acc = accuracy_score(y_test, y_pred_rf)
    rf_recall = recall_score(y_test, y_pred_rf)

    # -------------------
    # Print Comparison
    # -------------------
    print("\nModel Comparison\n")
    print("Logistic Regression:")
    print("  Accuracy:", round(log_acc, 3))
    print("  Recall:", round(log_recall, 3))
    print("  ROC-AUC:", round(log_auc, 3))

    print("\nRandom Forest:")
    print("  Accuracy:", round(rf_acc, 3))
    print("  Recall:", round(rf_recall, 3))
    print("  ROC-AUC:", round(rf_auc, 3))

    return rf_model, X.columns

def plot_feature_importance(model, feature_names):
    importances = model.feature_importances_

    indices = importances.argsort()[-10:]

    plt.figure(figsize=(8,6))
    plt.barh(range(len(indices)), importances[indices])
    plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
    plt.title("Top 10 Feature Importances (Random Forest)")
    plt.tight_layout()
    plt.savefig(os.path.join(os.path.dirname(os.path.dirname(__file__)), "results", "rf_feature_importance.png"))
    plt.show()

if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(BASE_DIR, "data", "telco_churn.csv")

    df = load_data(data_path)
    X, y = preprocess(df)

    rf_model, feature_names = train(X, y)
    plot_feature_importance(rf_model, feature_names)