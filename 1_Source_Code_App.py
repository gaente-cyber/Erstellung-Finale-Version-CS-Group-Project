import streamlit as st # Importieren von Streamlit für den Aufbau des App-Interface
import os # Importieren von os, um mit den richtigen Betriebssystemen zu interagieren
import json # Importieren von JSON, um JSON-Dateien lesen und schreiben zu können
import pandas as pd # Importieren von pandas, um Datenanalysen sowie auch die Verarbeitung von Tabellen zu bewerkstelligen
import matplotlib.pyplot as plt # zur Erstellung von Diagrammen und Darstellung von Analysen
import spotipy # um den Zugriff auf die Spotify Web API herstellen zu können
from spotipy.oauth2 import SpotifyOAuth, SpotifyClientCredentials # um den Zugriff auf die Spotify Web API herstellen zu können
from sklearn.feature_extraction.text import TfidfVectorizer # für die Textanalyse der Daten
from sklearn.metrics.pairwise import cosine_similarity # für Ähnlichkeitsvergleiche zwischen Texten
import logging # um systematisches Loggen von Informationen machen zu können (Fehlermeldungen oder Statusmeldungen etc.)

import joblib

# Lädt das vortrainierte Modell und den Binarizer
clf = joblib.load("clf.joblib")
mlb = joblib.load("mlb.joblib")

# -------------------- Einrichtung Logging System  --------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler("app.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# -------------------- Konfiguration von Streamlit --------------------
st.set_page_config(page_title="Spotify Podcast Filter", layout="centered") # Festlegung des Layouts der App

if "page_number_app" not in st.session_state:
    st.session_state.page_number_app = 0
if "user_profile" not in st.session_state:
    st.session_state.user_profile = {
        "interests": [],
        "languages": [],
        "duration": (20, 60),
        "mood": [],
        "location": ""
    }

profile_file = "user_profile.json" # Dateiname (profil_file), in dem das Nutzerprofil lokal gespeichert wird

# speichern der Benutzerdaten. Falls noch kein Profil besteht, wird hier die Grundlage, für die Möglichkeit ein Profil zu erstellen, geschaffen (Initialisierung der Listen etc.)

# -------------------- Spotify Authentifizierung --------------------
try:
    sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials(
    client_id="554c66803ecd4fe5ae23039953589266",
    client_secret="42c3865847ef4b1885776511a21290f4"
))
    
# Initialisierung der client-basierten Authentifizierung. Hiermit kann nun auf die öffentliche API zugegriffen werden

    logger.info("Spotify authentication successful.")
except Exception as e:
    logger.exception("Spotify authentication failed.")
    st.error("❌ Failed to authenticate with Spotify. Check logs for details.")
    st.stop()

# Falls die Authentifizierung fehl schlägt, wird über die obigen Schritte eine Fehlermeldung in Streamlit angezeigt, sowie auch die App mit der st.stop() Methode abgebrochen.

# -------------------- Definition der Schritte zur Navigierung der App --------------------
welcome_steps = ["Welcome", "Interests", "Languages", "Duration", "Mood", "Location"]

def next_page():
    st.session_state.page_number_app += 1

def previous_page():
    if st.session_state.page_number_app > 0:
        st.session_state.page_number_app -= 1

# Funktionen zur Navigierung in der App, Funktion, die eine Seite nach Vorne oder eine Seite nach Hinten geht

progress = min((st.session_state.page_number_app + 1) / len(welcome_steps), 1.0)
st.progress(progress)

# Zeigt in der Streamlit App einen Fortschrittsbalken (progress), der den aktuellen Fortschritt des Nutzers/Nutzerin in der Navigation der App anzeigt

# -------------------- Schritt 0: Begrüssung der Nutzerinnen in der Streamlit App --------------------
if st.session_state.page_number_app == 0:
    st.title("Welcome to Your Personalized Podcast Experience!")
    st.markdown("Let's set up your profile to get the best podcast recommendations!")
    if st.button("🚀 Get Started"):
        logger.info("User started onboarding.")
        next_page()

# Anzeigen des Begrüssungstitels und des Buttons "Get Started, der zur nächsten Seite führt

# -------------------- Schritt 1: Auswahl der Podcast Interessen --------------------
elif st.session_state.page_number_app == 1:
    st.title("🎯 Select Your Interests")
    spotify_podcast_interest_options = {
        "True Crime": "🕵️‍♂️", "Science": "🔬", "Comedy": "🎭", "Sports": "🏅",
        "Business": "💼", "Technology": "💻", "Society": "🏛️", "Health": "🧘",
        "Art": "🎨", "Music": "🎵", "Spirituality": "🧘‍♂️"
    }

# Hier werden die Interessen (im Dictionary "spotify_podcast_interest_options" gespeichert) der Nutzerinnen erörtert

    selected_interests = []
    cols = st.columns(3)
    for idx, (interest, emoji) in enumerate(spotify_podcast_interest_options.items()):
        with cols[idx % 3]:
            if st.checkbox(f"{emoji} {interest}"):
                selected.append(interest)

    st.session_state.user_profile["interests"] = selected_interests

# Vorbereitung: Erstellung der leeren Liste "selected"
# Darstellung: Auswahl der Interessen wird mittels Checkboxen in Streamlit dargestellts (in drei Spalten)
# Auswertung und Speichern: Anschliessend werden die ausgewählten Interessen der Nutzerinnen in der Liste "selected" gespeichert, sowie auch im "session_state" -> porträtiert den aktuellen Zustand der Sitzung 

# -------------------- Schritt 2: Auswahl der präferierten Sprachen --------------------
elif st.session_state.page_number_app == 2:
    st.title("🌍 Select Your Preferred Languages")
    language_options_app = {
        "German": "🇩🇪",
        "English": "🇬🇧",
    }

    selected_languages = []
    cols = st.columns(3)
    for idx, (language, flag) in enumerate(language_options_app.items()):
        with cols[idx % 3]:
            if st.checkbox(f"{flag} {language}"):
                selected_languages.append(language)

    st.session_state.user_profile["languages"] = selected_languages

# Sprachauswahl des Nutzerprofils und Speicherung der ausgewählten Sprachen
# Die Auswahl der Nutzerinnen wird unter "languages" gespeichert.

# -------------------- Schritt 3: Auswahl der Episodendauer der Podcasts --------------------
elif st.session_state.page_number_app == 3:
    st.title("🕒 Select Your Ideal Episode Length")
    st.session_state.user_profile["duration"] = st.slider(
        "Select the duration range (minutes)",
        min_value=5,
        max_value=120,
        value=st.session_state.user_profile["duration"]
    )

# Konfiguration des Sliders in Streamlit (interaktiver Regler zur Auswahl der Episodendauer) zur Auswahl der Episodendauer des Podcasts

# -------------------- Schritt 4: Auswahl der gewünschten Stimmung der Podcastauswahl --------------------
elif st.session_state.page_number_app == 4:
    st.title("😊 Choose the Mood of Podcasts")
    mood_options_app = [
        "Relaxation", "Learning", "Laughter",
        "Inspiration", "Focus", "Distraction"
    ]

    selected_moods_user = st.multiselect("Select one or more moods:", mood_options_app)
    st.session_state.user_profile["mood"] = selected_moods_user

# Mehrfachauswahl der gewünschten Stimmungen des Podcasts. 
# Die Auswahl der Nutzerinnen wird unter "mood" gespeichert.

# -------------------- Schritt 5: Auswahl des Ortes des Hörens des Podcasts --------------------
elif st.session_state.page_number_app == 5:
    st.title("📍 Where Do You Usually Listen?")
    st.session_state.user_profile["location"] = st.selectbox(
        "Select your main podcast listening location:",
        ["At home", "On the go", "In the car", "While exercising"]
    )

# Auswahl des Hörortes des Podcasts, wobei die Auswahl unter "location" gespeichert wird. 

# -------------------- Speichern des erstellten Nutzerprofils --------------------
    if st.button("✅ Save Profile & Start Podcast Filter"):
        try:
            with open(profile_file, "w") as f:
                json.dump(st.session_state.user_profile, f)
            logger.info(f"User profile saved: {st.session_state.user_profile}")
            st.success("🎉 Your profile has been saved! Starting the podcast filter...")
            next_page()
        except Exception as e:
            logger.exception("Failed to save user profile.")
            st.error("❌ Could not save profile.")

# der Button speichert das Nutzerprofil und startet die App
# Profil wird als JSON Datei geschrieben
# Anschliessend: Erfolgsmeldung und Weiterleitung auf die nächste Seite ODER Fehlerbehandlung und Logging (um Informationen über den Programmablauf zu sammeln und Fehlerbehandlung zu ermöglichen)

# -------------------- Navigations Buttons --------------------
if 0 < st.session_state.page_number_app < len(welcome_steps):
    col1, col2 = st.columns(2)
    with col1:
        st.button("⬅️ Back", on_click=previous_page)
    with col2:
        if st.session_state.page_number_app < len(welcome_steps) - 1:
            st.button("➡️ Next", on_click=next_page)

# Navigationsfunktion des Blocks: Zeigt die Navigations Buttons in der App. Die Buttons sind die "Zurück" und "Weiter"-Buttons. Mit diesen kann der Nutzer zwischen den Startseiten navigieren.

# -------------------- Hauptteil der App --------------------
if st.session_state.page_number_app >= len(welcome_steps):
    if os.path.exists(profile_file):
        try:
            with open(profile_file, "r") as f:
                user_profile = json.load(f)
            logger.info("Loaded user profile from file.")
        except Exception as e:
            logger.exception("Failed to load user profile.")
            st.error("❌ Failed to load saved profile.")
            st.stop()

        st.title("🎵 Intelligent Spotify Podcast Filter")
        st.markdown("### Your Personal Podcast Search")

        topics = ["True Crime", "Science", "Comedy", "Sports", "Business", "Health", "Technology", "Society"]
        selected_topics = st.multiselect("🎙️ Choose Podcast Topics", topics)
        language_filter = st.multiselect("🌍 Choose Language(s)", ["German", "English"])
        country_code = st.selectbox("🌐 Search in which country?", ["CH", "DE", "AT", "US", "GB"], index=0)
        host_name = st.text_input("👤 Host name (optional)")
        guest_name = st.text_input("🧑‍🤝‍🧑 Guest name(s) (optional)")
        min_duration, max_duration = st.slider("⏱️ Episode Duration (Minutes)", 5, 180, (10, 60))

        if st.button("🔍 Start Search"):
            logger.info("Search initiated.")
            logger.info(f"Filters: topics={selected_topics}, lang={language_filter}, country={country_code}, host={host_name}, guest={guest_name}, duration=({min_duration}, {max_duration})")

            profile_text = " ".join(user_profile.get("interests", []) + user_profile.get("mood", []))
            profile_text += " " + user_profile.get("location", "")
            profile_text += f" Duration {user_profile.get('duration', [0, 0])[0]} to {user_profile.get('duration', [0, 0])[1]} minutes"
            profile_text += " " + " ".join(selected_topics)
            profile_text += " " + host_name + " " + guest_name
            query = " ".join(user_profile.get("interests", []) + selected_topics)

            if not query.strip():
                st.warning("⚠️ Please select at least one interest or topic!")
            else:
                try:
                    with st.spinner("🔎 Searching..."):
                        results = sp.search(q=query, type="show", limit=20, market=country_code)
                        logger.info(f"Received {len(results['shows']['items'])} podcast shows.")
                        show_texts, show_info, average_duration, language_list = [], [], [], []

                        language_map = {"German": "de", "English": "en"}
                        allowed_language_codes = [language_map[l] for l in language_filter if l in language_map]

                        for show in results["shows"]["items"]:
                            name = show["name"]
                            description = show["description"]
                            publisher = show["publisher"]
                            show_language = show.get("languages", [""])[0].lower()
                            show_id = show["id"]
                            link = show["external_urls"]["spotify"]
                            image_url = show["images"][0]["url"] if show.get("images") else ""

                            if allowed_language_codes and show_language not in allowed_language_codes:
                                continue

                            episodes = sp.show_episodes(show_id, limit=3)["items"]
                            episodes_text = " ".join([ep["description"] for ep in episodes if ep and "description" in ep])
                            episodes_duration = [int(ep["duration_ms"] / 60000) for ep in episodes if ep and "duration_ms" in ep]

                            if not any(min_duration <= d <= max_duration for d in episodes_duration):
                                continue

                            full_text = description + " " + episodes_text
                            show_texts.append(full_text)
                            show_info.append((name, publisher, description, link, image_url))
                            if episodes_duration:
                                average_duration.append((name, sum(episodes_duration) / len(episodes_duration)))
                            language_list.append(show_language)

                        if show_texts:
                            texts = show_texts + [profile_text]
                            tfidf = TfidfVectorizer(stop_words='english')
                            matrix = tfidf.fit_transform(texts)
                            similarities = cosine_similarity(matrix[-1], matrix[:-1])[0]
                            top_indices = similarities.argsort()[-15:][::-1]

                            st.subheader("🎙️ Recommended Podcasts for You:")
                            for idx in top_indices:
                                name, publisher, description, link, image_url = show_info[idx]
                                st.image(image_url, width=200)
                                st.markdown(f"*🎧 {name}* – by {publisher}")
                                st.write(f"📝 {description[:250]}...")
                                st.markdown(f"[🔗 Go to podcast show on Spotify]({link})")
                                st.markdown("---")

                                                            # === 3 zusätzliche ML-Empfehlungen ===
                            # 1) Vektor aus dem Profil bauen
                            chosen = st.session_state.user_profile["interests"]
                            user_vec = mlb.transform([chosen]).toarray()[0]

                            # 2) Wahrscheinlichkeiten berechnen und sortieren
                            probs = clf.predict_proba(user_vec.reshape(1,-1))[0]
                            top_ml = probs.argsort()[::-1]

                            # 3) Bereits gezeigte Titel herausfiltern
                            shown = {show_info[i][0] for i in top_indices}  # Name aus show_info
                            extra = []
                            for i in top_ml:
                                title = mlb.classes_[i]
                                if title not in shown:
                                    extra.append((title, probs[i]))
                                if len(extra) == 3:
                                    break

                            # 4) Ausgabe
                            st.subheader("🎯 3 ML-basierte Zusatz-Empfehlungen")
                            for title, score in extra:
                                st.markdown(f"- **{title}** (Score: {score:.2f})")


                            # Visualisierung der Podcast Suchresultate
                            st.subheader("📊 Your Search Insights")
                            if average_duration:
                                logger.info("Plotting average episode durations.")
                                names, durations = zip(*average_duration)
                                fig, ax = plt.subplots()
                                ax.barh(names, durations, color="skyblue")
                                ax.set_xlabel("Minutes")
                                ax.set_title("Average Episode Duration")
                                ax.invert_yaxis()
                                st.pyplot(fig)

                            if language_list:
                                logger.info("Plotting language distribution.")
                                lang_counts = pd.Series(language_list).value_counts()
                                fig2, ax2 = plt.subplots()
                                lang_counts.plot(kind="bar", color="lightgreen", ax=ax2)
                                ax2.set_ylabel("Count")
                                ax2.set_title("Languages of Recommended Shows")
                                ax2.set_yticks(range(1, lang_counts.max() + 1))
                                st.pyplot(fig2)
                        else:
                            logger.warning("No matching podcasts found.")
                            st.warning("⚠️ No matching podcasts found. Please adjust your filters.")
                except Exception as e:
                    logger.exception("Error occurred during podcast search.")
                    st.error("❌ An error occurred during search. See logs for more details.")


