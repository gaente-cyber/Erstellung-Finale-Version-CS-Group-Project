## Voraussetzungen

- Python 3.8+
- Eine funktionierende Internetverbindung (für Spotify API)
- Zugriff auf die Spotify API Keys (`spotify_api_keys.py`)

## Installation

Installiere alle benötigten Abhängigkeiten mit:

```bash
pip install -r requirements.txt
```

## Modell trainieren

Bevor du die App startest, musst du einmalig das ML-Modell trainieren:

```bash
python model_training.py
```

Das Skript erstellt alle benötigten .joblib-Dateien im machine_learning/-Ordner.

## Anwendung starten

```bash
streamlit run .\app.py
```
Die Anwendung öffnet sich automatisch im Browser. Falls dies nicht der Fall ist kann die Anwendung über `http://localhost:8501` im Browser öffnen.


## Zurücksetzen des Nutzerprofils (optional)

Falls du das Onboarding erneut durchlaufen möchtest, lösche die folgende Datei: `user_profile.json`