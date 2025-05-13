import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, LabelEncoder #OneHotEncoder für Kategorische Spalten und LabelEncoder für Strings zu eindeutige Ganzzahlen
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, top_k_accuracy_score
import joblib

# 1. User Podcasts Beispieldaten laden
user_podcasts = pd.read_csv("data/user_podcasts.csv")


# 3. OneHot-Encoding für type + language (Für jeden Typ und Sprache von Podcasts wird eine eigene (binäre) Spalte erstellt)
onehot_encoder = OneHotEncoder(sparse_output=False)
encoded = onehot_encoder.fit_transform(user_podcasts[["type", "language"]])
encoded_data = pd.DataFrame(encoded, columns=onehot_encoder.get_feature_names_out(["type", "language"]))

# Encoder auf der Festplatte speichern
joblib.dump(onehot_encoder, "machine_learning/type_language_encoder.joblib")

# 3. Titel in numerische Klassen umwandeln (Label-Encoding)
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(user_podcasts["title"])  # z.B. "Serial" → 0, "Gemischtes Hack" → 1

# Encoder speichern für die Verwendung um später Titel wieder zurückzuwandeln
joblib.dump(label_encoder, "machine_learning/title_encoder.joblib")


# 4. Daten für das Training vorbereiten
X = encoded_data

# Train/Test-Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 6. Modell trainieren
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Modell abspeichern
joblib.dump(model, "machine_learning/regression_model.joblib")

# 7. Evaluation
# Alle bekannten Titel (vom LabelEncoder) 
all_labels = np.arange(len(label_encoder.classes_))


# Accuracy (Top 1 vorhergesagter Titel)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Top-1 Accuracy: {accuracy:.3f}")

# Top-5 Accuracy
# Für jede Zeile von X_test (z.B. "Comedy" + "German") wird überprüft, ob sich ein Titel der vorhergesagten Top 5 in dieser y_test Zeile befindet
top5 = top_k_accuracy_score(
    y_test,
    model.predict_proba(X_test),
    k=5,
    labels=all_labels
)
print(f"Top-5 Accuracy: {top5:.3f}")