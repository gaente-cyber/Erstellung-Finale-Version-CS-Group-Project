import streamlit as st
import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import logging
import joblib
import numpy as np
import spotify_api_keys as api_keys


USER_PROFILE_FILE = "user_profile.json"


# -------------------- Logging Setup --------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler("app.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# -------------------- Streamlit Config --------------------
st.set_page_config(page_title="Spotify Podcast Filter", layout="centered")

# -------------------- Load ML Model and Encoders --------------------
model = joblib.load("machine_learning/regression_model.joblib")
type_lang_encoder = joblib.load("machine_learning/type_language_encoder.joblib")
title_encoder = joblib.load("machine_learning/title_encoder.joblib")

# -------------------- Spotify Authentication (once) --------------------
@st.cache_resource
def authenticate_spotify():
    try:
        sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials(
            client_id=api_keys.CLIENT_ID,
            client_secret=api_keys.CLIENT_SECRET
        ))
        logger.info("Spotify authentication successful.")
        return sp
    except Exception as e:
        logger.exception("Spotify authentication failed.")
        st.error("❌ Failed to authenticate with Spotify.")
        st.stop()

sp = authenticate_spotify()

# -------------------- User Profile Management --------------------
def load_user_profile():
    if os.path.exists(USER_PROFILE_FILE):
        try:
            with open(USER_PROFILE_FILE, "r") as f:
                profile = json.load(f)
            logger.info("User profile loaded.")
            return profile
        except Exception as e:
            logger.exception("Error loading profile.")
    return {
        "interests": [],
        "languages": [],
        "duration": (20, 60),
        "mood": [],
        "location": ""
    }

def save_user_profile(profile):
    try:
        with open(USER_PROFILE_FILE, "w") as f:
            json.dump(profile, f)
        logger.info("User profile saved.")
    except Exception as e:
        logger.exception("Failed to save profile.")
        st.error("❌ Could not save profile.")


def show_spotify_title(title, sp):
    try:
        results = sp.search(q=title, type="show", limit=1)
        shows = results["shows"]["items"]
        if not shows:
            st.warning(f"🔍 No Spotify result found for '{title}'")
            return

        show = shows[0]
        name = show.get("name", "")
        publisher = show.get("publisher", "")
        description = show.get("description", "")
        link = show["external_urls"]["spotify"]
        image_url = show["images"][0]["url"] if show.get("images") else ""

        st.image(image_url, width=200)
        st.markdown(f"*🎧 {name}* – by {publisher}")
        st.write(f"📝 {description[:250]}...")
        st.markdown(f"[🔗 Listen on Spotify]({link})")
        st.markdown("---")

    except Exception as e:
        st.error(f"❌ Failed to retrieve '{title}' from Spotify.")
        logging.exception("Spotify title lookup failed.")


# -------------------- App Steps --------------------
welcome_steps = ["Welcome", "Interests", "Languages", "Duration", "Mood", "Location"]

def next_page():
    st.session_state.page_number_app += 1

def previous_page():
    st.session_state.page_number_app -= 1

# -------------------- Onboarding Pages --------------------
def show_onboarding():
    progress = min((st.session_state.page_number_app + 1) / len(welcome_steps), 1.0)
    st.progress(progress)

    if st.session_state.page_number_app == 0:
        st.title("Welcome to Your Personalized Podcast Experience!")
        st.markdown("Let's set up your profile to get the best podcast recommendations!")
        if st.button("🚀 Get Started"):
            next_page()

    elif st.session_state.page_number_app == 1:
        st.title("🎯 Select Your Interests")
        interest_options = {
            "True Crime": "🕵️‍♂️", "Science": "🔬", "Comedy": "🎭", "Sports": "🏅",
            "Business": "💼", "Technology": "💻", "Society": "🏛️", "Health": "🧘",
            "Art": "🎨", "Music": "🎵", "Spirituality": "🧘‍♂️"
        }
        selected = []
        cols = st.columns(3)
        for i, (label, emoji) in enumerate(interest_options.items()):
            with cols[i % 3]:
                if st.checkbox(f"{emoji} {label}"):
                    selected.append(label)
        st.session_state.user_profile["interests"] = selected

    elif st.session_state.page_number_app == 2:
        st.title("🌍 Select Your Preferred Languages")
        options = {"German": "🇩🇪", "English": "🇬🇧"}
        selected = []
        cols = st.columns(2)
        for i, (lang, flag) in enumerate(options.items()):
            with cols[i % 2]:
                if st.checkbox(f"{flag} {lang}"):
                    selected.append(lang)
        st.session_state.user_profile["languages"] = selected

    elif st.session_state.page_number_app == 3:
        st.title("🕒 Select Your Ideal Episode Length")
        st.session_state.user_profile["duration"] = st.slider(
            "Select duration range (minutes)",
            5, 120,
            value=st.session_state.user_profile.get("duration", (20, 60))
        )

    elif st.session_state.page_number_app == 4:
        st.title("😊 Choose the Mood of Podcasts")
        options = ["Relaxation", "Learning", "Laughter", "Inspiration", "Focus", "Distraction"]
        st.session_state.user_profile["mood"] = st.multiselect("Select mood(s):", options)

    elif st.session_state.page_number_app == 5:
        st.title("📍 Where Do You Usually Listen?")
        st.session_state.user_profile["location"] = st.selectbox(
            "Select your main listening location:",
            ["At home", "On the go", "In the car", "While exercising"]
        )
        if st.button("✅ Save Profile & Start"):
            save_user_profile(st.session_state.user_profile)
            st.success("🎉 Profile saved. Starting podcast filter...")
            next_page()

    if 0 < st.session_state.page_number_app < len(welcome_steps):
        col1, col2 = st.columns(2)
        with col1:
            st.button("⬅️ Back", on_click=previous_page)
        with col2:
            if st.session_state.page_number_app < len(welcome_steps) - 1:
                st.button("➡️ Next", on_click=next_page)

# -------------------- Hauptseite: Empfehlungen + Suche --------------------
def show_main_app():
    st.title("🎵 Intelligent Spotify Podcast Filter")
    st.markdown("### Your Personal Podcast Search")

    profile = st.session_state.user_profile

    topics = ["True Crime", "Science", "Comedy", "Sports", "Business", "Health", "Technology", "Society"]
    # Auswahl der Typen für die Suche mit voreingestellten Interessen (falls vorhanden)
    selected_topics = st.multiselect("🎙️ Choose Podcast Topics", topics, default=profile.get("interests", []))
    # Voreingestellte Sprachen aus dem Benutzerprofil
    language_filter = st.multiselect("🌍 Choose Language(s)", ["German", "English"], default=profile.get("languages", []))
    country_code = st.selectbox("🌐 Country", ["CH", "DE", "AT", "US", "GB"], index=0)
    host_name = st.text_input("👤 Host name (optional)")
    guest_name = st.text_input("🧑‍🤝‍🧑 Guest name(s) (optional)")
    min_dur, max_dur = st.slider("⏱️ Episode Duration", 5, 180, value=profile.get("duration", (20, 60))
)

    if st.button("🔍 Start Search"):
        profile_updated = False

        # Immer überschreiben mit aktueller Auswahl (auch wenn Items entfernt wurden)
        if profile.get("interests", []) != selected_topics:
            profile["interests"] = selected_topics
            profile_updated = True

        if profile.get("languages", []) != language_filter:
            profile["languages"] = language_filter
            profile_updated = True

        if profile.get("duration", (20, 60)) != (min_dur, max_dur):
            profile["duration"] = (min_dur, max_dur)
            profile_updated = True

        if profile_updated:
            save_user_profile(profile)
            st.success("📁 Profile updated and saved.")


        # Suchtext zusammenbauen
        query = " ".join(profile.get("interests", []) + profile.get("languages", []))
        if not query.strip():
            st.warning("⚠️ Please select at least one interest or topic!")
            return
        
        # Suchanfrage lesbarer machen
        readable_query = ", ".join(profile.get("interests", []) + profile.get("languages", []))
        st.write("🔍 Query:", readable_query)

        with st.spinner("🔎 Searching..."):
            try:
                # Anfrage an Spotify senden
                results = sp.search(q=query, type="show", limit=20, market=country_code)
                items = results["shows"]["items"]
                st.write(f"🔢 Raw results from Spotify: {len(items)} shows")

                show_info, show_texts, avg_durations, langs = [], [], [], []
                language_map = {"German": "de", "English": "en"}
                allowed_langs = [language_map.get(l) for l in language_filter if l in language_map]

                for show in items:
                    show_language = show.get("languages", [""])[0].lower()

                    # Sprachfilter toleranter gestalten: z. B. "de-de" soll auch "de" akzeptieren
                    if allowed_langs and show_language not in allowed_langs:
                        continue

                    show_id = show["id"]
                    try:
                        episodes_data = sp.show_episodes(show_id, limit=3, market=country_code)
                        episodes = episodes_data.get("items", [])
                    except Exception as e:
                        logger.warning(f"Could not fetch episodes for show ID: {show_id}")
                        continue  # diese Show überspringen
                    durations = [int(ep.get("duration_ms", 0) / 60000) for ep in episodes if ep]

                    # Wenn keine Episoden oder keine passende Dauer → skip
                    if not durations or not any(min_dur <= d <= max_dur for d in durations):
                        continue

                    ep_text = " ".join([ep.get("description", "") for ep in episodes if ep and "description" in ep])
                    full_text = show.get("description", "") + " " + ep_text
                    show_info.append((
                        show.get("name", ""),
                        show.get("publisher", ""),
                        show.get("description", ""),
                        show.get("external_urls", {}).get("spotify", ""),
                        show["images"][0]["url"] if show.get("images") else ""
                    ))
                    show_texts.append(full_text)
                    avg_durations.append((show.get("name", ""), np.mean(durations)))
                    langs.append(show_language)

                if not show_texts:
                    st.warning("⚠️ No matching podcasts found. Try different filters.")
                    return

                # Ähnlichkeitsvergleich
                profile_text = " ".join(profile["interests"] + profile["mood"]) + " " + profile["location"]
                texts = show_texts + [profile_text]
                tfidf = TfidfVectorizer(stop_words='english')
                matrix = tfidf.fit_transform(texts)
                sims = cosine_similarity(matrix[-1], matrix[:-1])[0]
                top_indices = sims.argsort()[-15:][::-1]

                st.subheader("🎙️ Recommended Podcasts for You:")
                for idx in top_indices:
                    name, publisher, desc, link, image = show_info[idx]
                    st.image(image, width=200)
                    st.markdown(f"*🎧 {name}* – by {publisher}")
                    st.write(f"📝 {desc[:250]}...")
                    st.markdown(f"[🔗 Listen on Spotify]({link})")
                    st.markdown("---")

                # === 3 ML-Empfehlungen ===
                st.subheader("🎯 3 ML-basierte Zusatz-Empfehlungen")

                if selected_topics and language_filter:
                    for lang in language_filter:
                        for typ in selected_topics:
                            df = pd.DataFrame([[typ, lang]], columns=["type", "language"])
                            try:
                                vec = type_lang_encoder.transform(df)
                                vec_df = pd.DataFrame(vec, columns=type_lang_encoder.get_feature_names_out(["type", "language"]))
                                probs = model.predict_proba(vec_df)[0]
                                top3 = np.argsort(probs)[-3:][::-1]
                                titles = title_encoder.inverse_transform(top3)

                                st.markdown(f"**Top 3 for '{typ}' in '{lang}':**")
                                for i, title in enumerate(titles):
                                    st.markdown(f"**🔢 Rank {i+1} (score: {probs[top3[i]]:.2f})**")
                                    show_spotify_title(title, sp)
                            except Exception as e:
                                st.warning(f"⚠️ Could not compute prediction for '{typ}' in '{lang}'")
                                logger.exception("Prediction error")
                else:
                    st.info("ℹ️ Please select at least one topic and one language to get ML-based recommendations.")

                # Visualisierung
                if avg_durations:
                    st.subheader("📊 Episode Duration Overview")
                    names, durations = zip(*avg_durations)
                    fig, ax = plt.subplots()
                    ax.barh(names, durations, color="skyblue")
                    ax.set_xlabel("Minutes")
                    ax.set_title("Average Episode Duration")
                    ax.invert_yaxis()
                    st.pyplot(fig)

                if langs:
                    lang_series = pd.Series(langs).value_counts()
                    fig2, ax2 = plt.subplots()
                    lang_series.plot(kind="bar", ax=ax2, color="lightgreen")
                    ax2.set_title("Languages of Recommendations")
                    st.pyplot(fig2)

            except Exception as e:
                logger.exception("Error in search.")
                st.error("❌ Search failed. Check logs for details.")

# -------------------- App Initialisierung --------------------
if "initialized" not in st.session_state:
    if os.path.exists(USER_PROFILE_FILE):
        st.session_state.user_profile = load_user_profile()
        st.session_state.page_number_app = len(welcome_steps)  # direkt zur Hauptansicht
    else:
        st.session_state.user_profile = {
            "interests": [],
            "languages": [],
            "duration": (20, 60),
            "mood": [],
            "location": ""
        }
        st.session_state.page_number_app = 0  # Onboarding starten

    st.session_state.initialized = True

# -------------------- App Routing --------------------
if st.session_state.page_number_app < len(welcome_steps):
    show_onboarding()
else:
    show_main_app()
