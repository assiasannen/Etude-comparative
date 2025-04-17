import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier, GradientBoostingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler

# charger le jeu de donnees
data = load_breast_cancer()
X = data.data
y = data.target

# normaliser les donnees (important pour la convergence du modele logistique)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# separation train/test
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# def les modeles
models = {
    "Bagging": BaggingClassifier(n_estimators=50, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "Boosting": GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=42),
    "Stacking": StackingClassifier(
        estimators=[
            ('lr', LogisticRegression(max_iter=1000)),
            ('svc', SVC(kernel='linear', probability=True))
        ],
        final_estimator=RandomForestClassifier(n_estimators=50, random_state=42)
    )
}

# fonction devaluation
def evaluate_model(name, model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return {
        "Model": name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1-score": f1_score(y_test, y_pred)
    }

# 6. Appliquer tous les modèles
results = []
for name, model in models.items():
    try:
        results.append(evaluate_model(name, model, X_train, y_train, X_test, y_test))
    except Exception as e:
        print(f"erreur avec le modèle {name} : {e}")

# affichage des resultats tabulaires
df_results = pd.DataFrame(results)
df_results = df_results.set_index("Model")
print("\n Résultats comparatifs :")
print(df_results.round(4))

# affichage graphique
df_results.plot(kind='bar', figsize=(10, 6))
plt.title('comparaison des performances des modeles')
plt.ylabel('Score')
plt.ylim(0.8, 1.0)
plt.grid(axis='y')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
