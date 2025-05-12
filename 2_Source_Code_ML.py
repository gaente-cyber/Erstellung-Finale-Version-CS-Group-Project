import numpy as np
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, top_k_accuracy_score

# 1. Deine Datenstruktur (alle Nutzer-IDs von 0 bis 179)
user_podcasts = [
    {'id': 0, 'p': [{'t': 'Wissenschaft im Brennpunkt', 'tp': 'Science', 'l': 'German'}, {'t': 'KUNST PAUSE', 'tp': 'Art', 'l': 'German'}, {'t': 'Gemischtes Hack', 'tp': 'Comedy', 'l': 'German'}]},
    {'id': 1, 'p': [{'t': 'Serial', 'tp': 'True Crime', 'l': 'English'}, {'t': 'The Vergecast', 'tp': 'Technology', 'l': 'English'}, {'t': 'Comedy Bang Bang', 'tp': 'Comedy', 'l': 'English'}]},
    {'id': 2, 'p': [{'t': 'The Art History Babes', 'tp': 'Art', 'l': 'English'}, {'t': 'Healthier Together', 'tp': 'Health', 'l': 'English'}, {'t': 'All Songs Considered', 'tp': 'Music', 'l': 'English'}]},
    {'id': 3, 'p': [{'t': 'Popcast', 'tp': 'Music', 'l': 'German'}, {'t': 'Deutschland3000', 'tp': 'Society', 'l': 'German'}]},
    {'id': 4, 'p': [{'t': 'CT Uplink', 'tp': 'Technology', 'l': 'German'}, {'t': 'Kicker meets DAZN', 'tp': 'Sports', 'l': 'German'}, {'t': 'Deutschland3000', 'tp': 'Society', 'l': 'German'}]},
    {'id': 5, 'p': [{'t': 'Waveform', 'tp': 'Technology', 'l': 'English'}, {'t': 'Hidden Brain', 'tp': 'Society', 'l': 'English'}]},
    {'id': 6, 'p': [{'t': 'My Favorite Murder', 'tp': 'True Crime', 'l': 'English'}, {'t': 'Comedy Bang Bang', 'tp': 'Comedy', 'l': 'English'}, {'t': 'Switched on Pop', 'tp': 'Music', 'l': 'English'}]},
    {'id': 7, 'p': [{'t': 'KUNST PAUSE', 'tp': 'Art', 'l': 'German'}, {'t': 'Buddhaweisheit', 'tp': 'Spirituality', 'l': 'German'}, {'t': 'Gemischtes Hack', 'tp': 'Comedy', 'l': 'German'}, {'t': 'Und was machst du am Wochenende?', 'tp': 'Sports', 'l': 'German'}]},
    {'id': 8, 'p': [{'t': 'KUNST DRUCK', 'tp': 'Art', 'l': 'German'}, {'t': 'IQ Wissenschaft und Forschung', 'tp': 'Science', 'l': 'German'}, {'t': 'Kicker meets DAZN', 'tp': 'Sports', 'l': 'German'}, {'t': 'Reflektor', 'tp': 'Music', 'l': 'German'}]},
    {'id': 9, 'p': [{'t': 'Oprah’s SuperSoul', 'tp': 'Spirituality', 'l': 'English'}, {'t': 'The Art History Babes', 'tp': 'Art', 'l': 'English'}]},
    {'id': 10, 'p': [{'t': 'How I Built This', 'tp': 'Business', 'l': 'English'}, {'t': 'Radiolab', 'tp': 'Science', 'l': 'English'}, {'t': 'ArtCurious', 'tp': 'Art', 'l': 'English'}]},
    {'id': 11, 'p': [{'t': 'Radiolab', 'tp': 'Science', 'l': 'English'}, {'t': 'Switched on Pop', 'tp': 'Music', 'l': 'English'}, {'t': 'ArtCurious', 'tp': 'Art', 'l': 'English'}]},
    {'id': 12, 'p': [{'t': 'OMR Podcast', 'tp': 'Business', 'l': 'German'}, {'t': 'Kicker meets DAZN', 'tp': 'Sports', 'l': 'German'}, {'t': 'KUNST DRUCK', 'tp': 'Art', 'l': 'German'}, {'t': 'Buddhaweisheit', 'tp': 'Spirituality', 'l': 'German'}]},
    {'id': 13, 'p': [{'t': 'Business Wars', 'tp': 'Business', 'l': 'English'}, {'t': 'The Ezra Klein Show', 'tp': 'Society', 'l': 'English'}]},
    {'id': 14, 'p': [{'t': 'All Songs Considered', 'tp': 'Music', 'l': 'English'}, {'t': 'Code Switch', 'tp': 'Society', 'l': 'English'}, {'t': 'The Infinite Monkey Cage', 'tp': 'Science', 'l': 'English'}, {'t': 'Talk Art', 'tp': 'Art', 'l': 'English'}]},
    {'id': 15, 'p': [{'t': 'KUNST DRUCK', 'tp': 'Art', 'l': 'German'}, {'t': 'Wissenschaft im Brennpunkt', 'tp': 'Science', 'l': 'German'}, {'t': 'Apotheken Umschau', 'tp': 'Health', 'l': 'German'}]},
    {'id': 16, 'p': [{'t': 'Radiolab', 'tp': 'Science', 'l': 'English'}, {'t': 'Serial', 'tp': 'True Crime', 'l': 'English'}]},
    {'id': 17, 'p': [{'t': 'Serial', 'tp': 'True Crime', 'l': 'English'}, {'t': 'The Infinite Monkey Cage', 'tp': 'Science', 'l': 'English'}, {'t': 'Hidden Brain', 'tp': 'Society', 'l': 'English'}]},
    {'id': 18, 'p': [{'t': 'Deutschland3000', 'tp': 'Society', 'l': 'German'}, {'t': 'brand eins Podcast', 'tp': 'Business', 'l': 'German'}, {'t': 'Lage der Liga', 'tp': 'Sports', 'l': 'German'}]},
    {'id': 19, 'p': [{'t': 'The Infinite Monkey Cage', 'tp': 'Science', 'l': 'English'}, {'t': 'Comedy Bang Bang', 'tp': 'Comedy', 'l': 'English'}, {'t': 'Song Exploder', 'tp': 'Music', 'l': 'English'}, {'t': 'Waveform', 'tp': 'Technology', 'l': 'English'}]},
    {'id': 20, 'p': [{'t': 'The Tim Ferriss Show', 'tp': 'Business', 'l': 'English'}, {'t': 'Reply All', 'tp': 'Technology', 'l': 'English'}, {'t': 'Hidden Brain', 'tp': 'Society', 'l': 'English'}, {'t': 'The Bill Simmons Podcast', 'tp': 'Sports', 'l': 'English'}]},
    {'id': 21, 'p': [{'t': 'Logbuch:Netzpolitik', 'tp': 'Technology', 'l': 'German'}, {'t': 'Und was machst du am Wochenende?', 'tp': 'Sports', 'l': 'German'}, {'t': 'brand eins Podcast', 'tp': 'Business', 'l': 'German'}]},
    {'id': 22, 'p': [{'t': 'Die sogenannte Gegenwart', 'tp': 'Society', 'l': 'German'}, {'t': 'Baywatch Berlin', 'tp': 'Comedy', 'l': 'German'}, {'t': 'KUNST PAUSE', 'tp': 'Art', 'l': 'German'}, {'t': 'Logbuch:Netzpolitik', 'tp': 'Technology', 'l': 'German'}]},
    {'id': 23, 'p': [{'t': 'My Dad Wrote A Porno', 'tp': 'Comedy', 'l': 'English'}, {'t': 'All Songs Considered', 'tp': 'Music', 'l': 'English'}, {'t': 'How I Built This', 'tp': 'Business', 'l': 'English'}, {'t': 'The Art History Babes', 'tp': 'Art', 'l': 'English'}, {'t': 'Reply All', 'tp': 'Technology', 'l': 'English'}]},
    {'id': 24, 'p': [{'t': 'Switched on Pop', 'tp': 'Music', 'l': 'English'}, {'t': 'Pardon My Take', 'tp': 'Sports', 'l': 'English'}, {'t': 'My Favorite Murder', 'tp': 'True Crime', 'l': 'English'}, {'t': 'The Tim Ferriss Show', 'tp': 'Business', 'l': 'English'}]},
    {'id': 25, 'p': [{'t': 'My Dad Wrote A Porno', 'tp': 'Comedy', 'l': 'English'}, {'t': 'Switched on Pop', 'tp': 'Music', 'l': 'English'}]},
    {'id': 26, 'p': [{'t': 'A New Earth', 'tp': 'Spirituality', 'l': 'English'}, {'t': 'Science Vs', 'tp': 'Science', 'l': 'English'}, {'t': 'Song Exploder', 'tp': 'Music', 'l': 'English'}]},
    {'id': 27, 'p': [{'t': 'How I Built This', 'tp': 'Business', 'l': 'English'}, {'t': 'My Favorite Murder', 'tp': 'True Crime', 'l': 'English'}, {'t': 'Feel Better Live More', 'tp': 'Health', 'l': 'English'}, {'t': 'The Ezra Klein Show', 'tp': 'Society', 'l': 'English'}]},
    {'id': 28, 'p': [{'t': 'Herzzeit', 'tp': 'Spirituality', 'l': 'German'}, {'t': 'Handelsblatt Today', 'tp': 'Business', 'l': 'German'}, {'t': 'Reflektor', 'tp': 'Music', 'l': 'German'}]},
    {'id': 29, 'p': [{'t': 'Crime Junkie', 'tp': 'True Crime', 'l': 'English'}, {'t': 'The Vergecast', 'tp': 'Technology', 'l': 'English'}]},
    {'id': 30, 'p': [{'t': 'The Vergecast', 'tp': 'Technology', 'l': 'English'}, {'t': 'Switched on Pop', 'tp': 'Music', 'l': 'English'}, {'t': 'Code Switch', 'tp': 'Society', 'l': 'English'}, {'t': 'How I Built This', 'tp': 'Business', 'l': 'English'}]},
    {'id': 31, 'p': [{'t': 'Waveform', 'tp': 'Technology', 'l': 'English'}, {'t': 'Serial', 'tp': 'True Crime', 'l': 'English'}, {'t': 'Business Wars', 'tp': 'Business', 'l': 'English'}]},
    {'id': 32, 'p': [{'t': 'A New Earth', 'tp': 'Spirituality', 'l': 'English'}, {'t': 'Science Vs', 'tp': 'Science', 'l': 'English'}, {'t': 'Talk Art', 'tp': 'Art', 'l': 'English'}, {'t': 'Serial', 'tp': 'True Crime', 'l': 'English'}, {'t': "The Doctor's Farmacy", 'tp': 'Health', 'l': 'English'}]},
    {'id': 33, 'p': [{'t': 'All Songs Considered', 'tp': 'Music', 'l': 'English'}, {'t': 'Football Weekly', 'tp': 'Sports', 'l': 'English'}, {'t': 'How I Built This', 'tp': 'Business', 'l': 'English'}]},
    {'id': 34, 'p': [{'t': "The Doctor's Farmacy", 'tp': 'Health', 'l': 'English'}, {'t': 'Talk Art', 'tp': 'Art', 'l': 'English'}, {'t': 'Song Exploder', 'tp': 'Music', 'l': 'English'}]},
    {'id': 35, 'p': [{'t': 'All Songs Considered', 'tp': 'Music', 'l': 'English'}, {'t': 'The Infinite Monkey Cage', 'tp': 'Science', 'l': 'English'}, {'t': 'Oprah’s SuperSoul', 'tp': 'Spirituality', 'l': 'English'}, {'t': 'Business Wars', 'tp': 'Business', 'l': 'English'}]},
    {'id': 36, 'p': [{'t': 'Einmal täglich Glück', 'tp': 'Health', 'l': 'German'}, {'t': 'Popcast', 'tp': 'Music', 'l': 'German'}, {'t': 'CT Uplink', 'tp': 'Technology', 'l': 'German'}]},
    {'id': 37, 'p': [{'t': 'The Art History Babes', 'tp': 'Art', 'l': 'English'}, {'t': 'The Vergecast', 'tp': 'Technology', 'l': 'English'}, {'t': 'Football Weekly', 'tp': 'Sports', 'l': 'English'}]},
    {'id': 38, 'p': [{'t': 'Gesund Leben', 'tp': 'Health', 'l': 'German'}, {'t': 'Herzzeit', 'tp': 'Spirituality', 'l': 'German'}]},
    {'id': 39, 'p': [{'t': 'Reflektor', 'tp': 'Music', 'l': 'German'}, {'t': 'Herzzeit', 'tp': 'Spirituality', 'l': 'German'}]},
    {'id': 40, 'p': [{'t': 'Cosmo – Machiavelli', 'tp': 'Society', 'l': 'German'}, {'t': 'Wissenschaft im Brennpunkt', 'tp': 'Science', 'l': 'German'}, {'t': 'Fest & Flauschig', 'tp': 'Comedy', 'l': 'German'}, {'t': 'brand eins Podcast', 'tp': 'Business', 'l': 'German'}]},
    {'id': 41, 'p': [{'t': 'Waveform', 'tp': 'Technology', 'l': 'English'}, {'t': 'Healthier Together', 'tp': 'Health', 'l': 'English'}, {'t': 'The Daily Show: Ears Edition', 'tp': 'Comedy', 'l': 'English'}]},
    {'id': 42, 'p': [{'t': 'Cosmo – Machiavelli', 'tp': 'Society', 'l': 'German'}, {'t': 'Mordlust', 'tp': 'True Crime', 'l': 'German'}, {'t': 'KUNST DRUCK', 'tp': 'Art', 'l': 'German'}, {'t': 'OMR Podcast', 'tp': 'Business', 'l': 'German'}, {'t': 'Einmal täglich Glück', 'tp': 'Health', 'l': 'German'}]},
    {'id': 43, 'p': [{'t': 'Fest & Flauschig', 'tp': 'Comedy', 'l': 'German'}, {'t': 'Handelsblatt Today', 'tp': 'Business', 'l': 'German'}, {'t': 'Verbrechen', 'tp': 'True Crime', 'l': 'German'}, {'t': 'SWR2 Wissen', 'tp': 'Science', 'l': 'German'}]},
    {'id': 44, 'p': [{'t': 'Cosmo – Machiavelli', 'tp': 'Society', 'l': 'German'}, {'t': 'Buddhaweisheit', 'tp': 'Spirituality', 'l': 'German'}]},
    {'id': 45, 'p': [{'t': 'The Vergecast', 'tp': 'Technology', 'l': 'English'}, {'t': 'Radiolab', 'tp': 'Science', 'l': 'English'}]},
    {'id': 46, 'p': [{'t': 'The Daily Show: Ears Edition', 'tp': 'Comedy', 'l': 'English'}, {'t': 'The Tim Ferriss Show', 'tp': 'Business', 'l': 'English'}]},
    {'id': 47, 'p': [{'t': 'All Songs Considered', 'tp': 'Music', 'l': 'English'}, {'t': 'Business Wars', 'tp': 'Business', 'l': 'English'}, {'t': 'Hidden Brain', 'tp': 'Society', 'l': 'English'}, {'t': 'Feel Better Live More', 'tp': 'Health', 'l': 'English'}]},
    {'id': 48, 'p': [{'t': 'Oprah’s SuperSoul', 'tp': 'Spirituality', 'l': 'English'}, {'t': 'Football Weekly', 'tp': 'Sports', 'l': 'English'}, {'t': 'Science Vs', 'tp': 'Science', 'l': 'English'}]},
    {'id': 49, 'p': [{'t': 'On Being', 'tp': 'Spirituality', 'l': 'English'}, {'t': 'Comedy Bang Bang', 'tp': 'Comedy', 'l': 'English'}, {'t': 'The Bill Simmons Podcast', 'tp': 'Sports', 'l': 'English'}, {'t': 'The Vergecast', 'tp': 'Technology', 'l': 'English'}]},
    {'id': 50, 'p': [{'t': 'Science Vs', 'tp': 'Science', 'l': 'English'}, {'t': 'Talk Art', 'tp': 'Art', 'l': 'English'}, {'t': 'On Being', 'tp': 'Spirituality', 'l': 'English'}, {'t': 'Pardon My Take', 'tp': 'Sports', 'l': 'English'}]},
    {'id': 51, 'p': [{'t': 'The Daily Show: Ears Edition', 'tp': 'Comedy', 'l': 'English'}, {'t': 'Code Switch', 'tp': 'Society', 'l': 'English'}]},
    {'id': 52, 'p': [{'t': 'Comedy Bang Bang', 'tp': 'Comedy', 'l': 'English'}, {'t': 'Pardon My Take', 'tp': 'Sports', 'l': 'English'}, {'t': 'Morbid', 'tp': 'True Crime', 'l': 'English'}, {'t': 'Reply All', 'tp': 'Technology', 'l': 'English'}]},
    {'id': 53, 'p': [{'t': 'Der kreative Flow', 'tp': 'Art', 'l': 'German'}, {'t': 'Der spirituelle Podcast', 'tp': 'Spirituality', 'l': 'German'}]},
    {'id': 54, 'p': [{'t': 'My Favorite Murder', 'tp': 'True Crime', 'l': 'English'}, {'t': 'The Art History Babes', 'tp': 'Art', 'l': 'English'}, {'t': 'The Vergecast', 'tp': 'Technology', 'l': 'English'}, {'t': 'Hidden Brain', 'tp': 'Society', 'l': 'English'}]},
    {'id': 55, 'p': [{'t': 'CT Uplink', 'tp': 'Technology', 'l': 'German'}, {'t': 'Und was machst du am Wochenende?', 'tp': 'Sports', 'l': 'German'}, {'t': 'Einmal täglich Glück', 'tp': 'Health', 'l': 'German'}, {'t': 'Buddhaweisheit', 'tp': 'Spirituality', 'l': 'German'}, {'t': 'KUNST PAUSE', 'tp': 'Art', 'l': 'German'}]},
    {'id': 56, 'p': [{'t': 'Song Exploder', 'tp': 'Music', 'l': 'English'}, {'t': 'The Vergecast', 'tp': 'Technology', 'l': 'English'}, {'t': "The Doctor's Farmacy", 'tp': 'Health', 'l': 'English'}, {'t': 'Oprah’s SuperSoul', 'tp': 'Spirituality', 'l': 'English'}, {'t': 'The Art History Babes', 'tp': 'Art', 'l': 'English'}]},
    {'id': 57, 'p': [{'t': 'Pardon My Take', 'tp': 'Sports', 'l': 'English'}, {'t': 'The Art History Babes', 'tp': 'Art', 'l': 'English'}, {'t': 'Healthier Together', 'tp': 'Health', 'l': 'English'}, {'t': 'The Ezra Klein Show', 'tp': 'Society', 'l': 'English'}]},
    {'id': 58, 'p': [{'t': 'Business Wars', 'tp': 'Business', 'l': 'English'}, {'t': 'Waveform', 'tp': 'Technology', 'l': 'English'}]},
    {'id': 59, 'p': [{'t': 'ArtCurious', 'tp': 'Art', 'l': 'English'}, {'t': 'Song Exploder', 'tp': 'Music', 'l': 'English'}, {'t': 'The Infinite Monkey Cage', 'tp': 'Science', 'l': 'English'}, {'t': 'Business Wars', 'tp': 'Business', 'l': 'English'}, {'t': 'Hidden Brain', 'tp': 'Society', 'l': 'English'}]},
    {'id': 60, 'p': [{'t': 'Healthier Together', 'tp': 'Health', 'l': 'English'}, {'t': 'How I Built This', 'tp': 'Business', 'l': 'English'}, {'t': 'My Dad Wrote A Porno', 'tp': 'Comedy', 'l': 'English'}]},
    {'id': 61, 'p': [{'t': 'Cosmo – Machiavelli', 'tp': 'Society', 'l': 'German'}, {'t': 'KUNST PAUSE', 'tp': 'Art', 'l': 'German'}, {'t': 'Kicker meets DAZN', 'tp': 'Sports', 'l': 'German'}, {'t': 'Wissenschaft im Brennpunkt', 'tp': 'Science', 'l': 'German'}, {'t': 'Reflektor', 'tp': 'Music', 'l': 'German'}]},
    {'id': 62, 'p': [{'t': 'Technik und mehr', 'tp': 'Technology', 'l': 'German'}, {'t': 'KUNST DRUCK', 'tp': 'Art', 'l': 'German'}, {'t': 'Deutschland3000', 'tp': 'Society', 'l': 'German'}, {'t': 'Gesund Leben', 'tp': 'Health', 'l': 'German'}, {'t': 'SWR2 Wissen', 'tp': 'Science', 'l': 'German'}]},
    {'id': 63, 'p': [{'t': 'brand eins Podcast', 'tp': 'Business', 'l': 'German'}, {'t': 'Buddhaweisheit', 'tp': 'Spirituality', 'l': 'German'}]},
    {'id': 64, 'p': [{'t': 'Talk Art', 'tp': 'Art', 'l': 'English'}, {'t': 'The Daily Show: Ears Edition', 'tp': 'Comedy', 'l': 'English'}, {'t': 'Football Weekly', 'tp': 'Sports', 'l': 'English'}, {'t': 'Radiolab', 'tp': 'Science', 'l': 'English'}]},
    {'id': 65, 'p': [{'t': 'OMR Podcast', 'tp': 'Business', 'l': 'German'}, {'t': 'Buddhaweisheit', 'tp': 'Spirituality', 'l': 'German'}, {'t': 'Verbrechen', 'tp': 'True Crime', 'l': 'German'}]},
    {'id': 66, 'p': [{'t': 'Talk Art', 'tp': 'Art', 'l': 'English'}, {'t': 'Healthier Together', 'tp': 'Health', 'l': 'English'}, {'t': 'Serial', 'tp': 'True Crime', 'l': 'English'}, {'t': 'The Daily Show: Ears Edition', 'tp': 'Comedy', 'l': 'English'}]},
    {'id': 67, 'p': [{'t': 'Morbid', 'tp': 'True Crime', 'l': 'English'}, {'t': 'The Bill Simmons Podcast', 'tp': 'Sports', 'l': 'English'}, {'t': 'The Tim Ferriss Show', 'tp': 'Business', 'l': 'English'}, {'t': 'Comedy Bang Bang', 'tp': 'Comedy', 'l': 'English'}, {'t': 'The Ezra Klein Show', 'tp': 'Society', 'l': 'English'}]},
    {'id': 68, 'p': [{'t': 'KUNST DRUCK', 'tp': 'Art', 'l': 'German'}, {'t': 'Deutschland3000', 'tp': 'Society', 'l': 'German'}]},
    {'id': 69, 'p': [{'t': 'The Art History Babes', 'tp': 'Art', 'l': 'English'}, {'t': 'Serial', 'tp': 'True Crime', 'l': 'English'}, {'t': 'The Vergecast', 'tp': 'Technology', 'l': 'English'}]},
    {'id': 70, 'p': [{'t': 'The Ezra Klein Show', 'tp': 'Society', 'l': 'English'}, {'t': 'The Bill Simmons Podcast', 'tp': 'Sports', 'l': 'English'}, {'t': 'Waveform', 'tp': 'Technology', 'l': 'English'}]},
    {'id': 71, 'p': [{'t': 'Mordlust', 'tp': 'True Crime', 'l': 'German'}, {'t': 'Der kreative Flow', 'tp': 'Art', 'l': 'German'}]},
    {'id': 72, 'p': [{'t': 'The Art History Babes', 'tp': 'Art', 'l': 'English'}, {'t': 'Comedy Bang Bang', 'tp': 'Comedy', 'l': 'English'}, {'t': 'Code Switch', 'tp': 'Society', 'l': 'English'}, {'t': 'Song Exploder', 'tp': 'Music', 'l': 'English'}, {'t': 'Radiolab', 'tp': 'Science', 'l': 'English'}]},
    {'id': 73, 'p': [{'t': 'Oprah’s SuperSoul', 'tp': 'Spirituality', 'l': 'English'}, {'t': 'Crime Junkie', 'tp': 'True Crime', 'l': 'English'}, {'t': 'Business Wars', 'tp': 'Business', 'l': 'English'}]},
    {'id': 74, 'p': [{'t': 'Mordlust', 'tp': 'True Crime', 'l': 'German'}, {'t': 'Popcast', 'tp': 'Music', 'l': 'German'}, {'t': 'IQ Wissenschaft und Forschung', 'tp': 'Science', 'l': 'German'}]},
    {'id': 75, 'p': [{'t': 'The Vergecast', 'tp': 'Technology', 'l': 'English'}, {'t': 'The Daily Show: Ears Edition', 'tp': 'Comedy', 'l': 'English'}, {'t': 'Serial', 'tp': 'True Crime', 'l': 'English'}, {'t': 'ArtCurious', 'tp': 'Art', 'l': 'English'}, {'t': 'A New Earth', 'tp': 'Spirituality', 'l': 'English'}]},
    {'id': 76, 'p': [{'t': 'Reflektor', 'tp': 'Music', 'l': 'German'}, {'t': 'Verbrechen', 'tp': 'True Crime', 'l': 'German'}, {'t': 'Handelsblatt Today', 'tp': 'Business', 'l': 'German'}, {'t': 'KUNST PAUSE', 'tp': 'Art', 'l': 'German'}, {'t': 'Gemischtes Hack', 'tp': 'Comedy', 'l': 'German'}]},
    {'id': 77, 'p': [{'t': 'Feel Better Live More', 'tp': 'Health', 'l': 'English'}, {'t': 'Radiolab', 'tp': 'Science', 'l': 'English'}, {'t': 'Serial', 'tp': 'True Crime', 'l': 'English'}]},
    {'id': 78, 'p': [{'t': 'ArtCurious', 'tp': 'Art', 'l': 'English'}, {'t': 'My Favorite Murder', 'tp': 'True Crime', 'l': 'English'}, {'t': 'Hidden Brain', 'tp': 'Society', 'l': 'English'}]},
    {'id': 79, 'p': [{'t': 'Der spirituelle Podcast', 'tp': 'Spirituality', 'l': 'German'}, {'t': 'Wissenschaft im Brennpunkt', 'tp': 'Science', 'l': 'German'}]},
    {'id': 80, 'p': [{'t': 'On Being', 'tp': 'Spirituality', 'l': 'English'}, {'t': 'Hidden Brain', 'tp': 'Society', 'l': 'English'}, {'t': 'Healthier Together', 'tp': 'Health', 'l': 'English'}]},
    {'id': 81, 'p': [{'t': 'The Tim Ferriss Show', 'tp': 'Business', 'l': 'English'}, {'t': 'Science Vs', 'tp': 'Science', 'l': 'English'}]},
    {'id': 82, 'p': [{'t': 'All Songs Considered', 'tp': 'Music', 'l': 'English'}, {'t': 'How I Built This', 'tp': 'Business', 'l': 'English'}, {'t': 'On Being', 'tp': 'Spirituality', 'l': 'English'}]},
    {'id': 83, 'p': [{'t': 'Reply All', 'tp': 'Technology', 'l': 'English'}, {'t': 'On Being', 'tp': 'Spirituality', 'l': 'English'}, {'t': 'The Ezra Klein Show', 'tp': 'Society', 'l': 'English'}]},
    {'id': 84, 'p': [{'t': 'Code Switch', 'tp': 'Society', 'l': 'English'}, {'t': 'The Infinite Monkey Cage', 'tp': 'Science', 'l': 'English'}, {'t': 'Healthier Together', 'tp': 'Health', 'l': 'English'}, {'t': 'Reply All', 'tp': 'Technology', 'l': 'English'}, {'t': 'Talk Art', 'tp': 'Art', 'l': 'English'}]},
    {'id': 85, 'p': [{'t': 'Lage der Liga', 'tp': 'Sports', 'l': 'German'}, {'t': 'CT Uplink', 'tp': 'Technology', 'l': 'German'}, {'t': 'Herzzeit', 'tp': 'Spirituality', 'l': 'German'}]},
    {'id': 86, 'p': [{'t': 'Crime Junkie', 'tp': 'True Crime', 'l': 'English'}, {'t': 'Business Wars', 'tp': 'Business', 'l': 'English'}, {'t': 'The Art History Babes', 'tp': 'Art', 'l': 'English'}, {'t': 'Science Vs', 'tp': 'Science', 'l': 'English'}, {'t': 'Feel Better Live More', 'tp': 'Health', 'l': 'English'}]},
    {'id': 87, 'p': [{'t': 'Talk Art', 'tp': 'Art', 'l': 'English'}, {'t': 'The Vergecast', 'tp': 'Technology', 'l': 'English'}, {'t': 'The Bill Simmons Podcast', 'tp': 'Sports', 'l': 'English'}, {'t': 'The Ezra Klein Show', 'tp': 'Society', 'l': 'English'}]},
    {'id': 88, 'p': [{'t': 'ArtCurious', 'tp': 'Art', 'l': 'English'}, {'t': 'Hidden Brain', 'tp': 'Society', 'l': 'English'}]},
    {'id': 89, 'p': [{'t': 'Der spirituelle Podcast', 'tp': 'Spirituality', 'l': 'German'}, {'t': 'Die sogenannte Gegenwart', 'tp': 'Society', 'l': 'German'}, {'t': 'Technik und mehr', 'tp': 'Technology', 'l': 'German'}]},
    {'id': 90, 'p': [{'t': 'Comedy Bang Bang', 'tp': 'Comedy', 'l': 'English'}, {'t': 'How I Built This', 'tp': 'Business', 'l': 'English'}, {'t': 'Code Switch', 'tp': 'Society', 'l': 'English'}, {'t': 'On Being', 'tp': 'Spirituality', 'l': 'English'}, {'t': 'Talk Art', 'tp': 'Art', 'l': 'English'}]},
    {'id': 91, 'p': [{'t': 'Code Switch', 'tp': 'Society', 'l': 'English'}, {'t': 'Waveform', 'tp': 'Technology', 'l': 'English'}, {'t': 'Talk Art', 'tp': 'Art', 'l': 'English'}, {'t': "The Doctor's Farmacy", 'tp': 'Health', 'l': 'English'}]},
    {'id': 92, 'p': [{'t': 'The Tim Ferriss Show', 'tp': 'Business', 'l': 'English'}, {'t': 'Reply All', 'tp': 'Technology', 'l': 'English'}, {'t': 'The Ezra Klein Show', 'tp': 'Society', 'l': 'English'}, {'t': 'Oprah’s SuperSoul', 'tp': 'Spirituality', 'l': 'English'}]},
    {'id': 93, 'p': [{'t': 'Crime Junkie', 'tp': 'True Crime', 'l': 'English'}, {'t': 'Radiolab', 'tp': 'Science', 'l': 'English'}, {'t': 'How I Built This', 'tp': 'Business', 'l': 'English'}]},
    {'id': 94, 'p': [{'t': 'A New Earth', 'tp': 'Spirituality', 'l': 'English'}, {'t': 'Feel Better Live More', 'tp': 'Health', 'l': 'English'}, {'t': 'The Ezra Klein Show', 'tp': 'Society', 'l': 'English'}]},
    {'id': 95, 'p': [{'t': 'Football Weekly', 'tp': 'Sports', 'l': 'English'}, {'t': "The Doctor's Farmacy", 'tp': 'Health', 'l': 'English'}, {'t': 'The Art History Babes', 'tp': 'Art', 'l': 'English'}, {'t': 'Radiolab', 'tp': 'Science', 'l': 'English'}]},
    {'id': 96, 'p': [{'t': 'How I Built This', 'tp': 'Business', 'l': 'English'}, {'t': 'My Favorite Murder', 'tp': 'True Crime', 'l': 'English'}]},
    {'id': 97, 'p': [{'t': 'Switched on Pop', 'tp': 'Music', 'l': 'English'}, {'t': 'Reply All', 'tp': 'Technology', 'l': 'English'}, {'t': 'Comedy Bang Bang', 'tp': 'Comedy', 'l': 'English'}]},
    {'id': 98, 'p': [{'t': 'Comedy Bang Bang', 'tp': 'Comedy', 'l': 'English'}, {'t': 'A New Earth', 'tp': 'Spirituality', 'l': 'English'}, {'t': 'Football Weekly', 'tp': 'Sports', 'l': 'English'}, {'t': 'Code Switch', 'tp': 'Society', 'l': 'English'}, {'t': 'ArtCurious', 'tp': 'Art', 'l': 'English'}]},
    {'id': 99, 'p': [{'t': 'Talk Art', 'tp': 'Art', 'l': 'English'}, {'t': 'Comedy Bang Bang', 'tp': 'Comedy', 'l': 'English'}]},
    {'id': 100, 'p': [{'t': 'Lage der Liga', 'tp': 'Sports', 'l': 'German'}, {'t': 'Der spirituelle Podcast', 'tp': 'Spirituality', 'l': 'German'}, {'t': 'Logbuch:Netzpolitik', 'tp': 'Technology', 'l': 'German'}, {'t': 'OMR Podcast', 'tp': 'Business', 'l': 'German'}, {'t': 'Fest & Flauschig', 'tp': 'Comedy', 'l': 'German'}]},
    {'id': 101, 'p': [{'t': 'Lage der Liga', 'tp': 'Sports', 'l': 'German'}, {'t': 'Fest & Flauschig', 'tp': 'Comedy', 'l': 'German'}, {'t': 'Technik und mehr', 'tp': 'Technology', 'l': 'German'}, {'t': 'Der kreative Flow', 'tp': 'Art', 'l': 'German'}]},
    {'id': 102, 'p': [{'t': 'Waveform', 'tp': 'Technology', 'l': 'English'}, {'t': 'Business Wars', 'tp': 'Business', 'l': 'English'}, {'t': 'Science Vs', 'tp': 'Science', 'l': 'English'}, {'t': 'My Dad Wrote A Porno', 'tp': 'Comedy', 'l': 'English'}, {'t': 'My Favorite Murder', 'tp': 'True Crime', 'l': 'English'}]},
    {'id': 103, 'p': [{'t': 'Code Switch', 'tp': 'Society', 'l': 'English'}, {'t': 'The Art History Babes', 'tp': 'Art', 'l': 'English'}, {'t': 'Morbid', 'tp': 'True Crime', 'l': 'English'}, {'t': 'The Infinite Monkey Cage', 'tp': 'Science', 'l': 'English'}]},
    {'id': 104, 'p': [{'t': 'Reply All', 'tp': 'Technology', 'l': 'English'}, {'t': 'My Dad Wrote A Porno', 'tp': 'Comedy', 'l': 'English'}]},
    {'id': 105, 'p': [{'t': 'Radiolab', 'tp': 'Science', 'l': 'English'}, {'t': 'Code Switch', 'tp': 'Society', 'l': 'English'}, {'t': 'Talk Art', 'tp': 'Art', 'l': 'English'}, {'t': 'Reply All', 'tp': 'Technology', 'l': 'English'}, {'t': 'Business Wars', 'tp': 'Business', 'l': 'English'}]},
    {'id': 106, 'p': [{'t': 'The Tim Ferriss Show', 'tp': 'Business', 'l': 'English'}, {'t': 'Serial', 'tp': 'True Crime', 'l': 'English'}, {'t': "The Doctor's Farmacy", 'tp': 'Health', 'l': 'English'}]},
    {'id': 107, 'p': [{'t': 'Switched on Pop', 'tp': 'Music', 'l': 'English'}, {'t': 'The Infinite Monkey Cage', 'tp': 'Science', 'l': 'English'}, {'t': 'Feel Better Live More', 'tp': 'Health', 'l': 'English'}, {'t': 'On Being', 'tp': 'Spirituality', 'l': 'English'}]},
    {'id': 108, 'p': [{'t': 'Und was machst du am Wochenende?', 'tp': 'Sports', 'l': 'German'}, {'t': 'Baywatch Berlin', 'tp': 'Comedy', 'l': 'German'}]},
    {'id': 109, 'p': [{'t': 'The Tim Ferriss Show', 'tp': 'Business', 'l': 'English'}, {'t': 'Morbid', 'tp': 'True Crime', 'l': 'English'}, {'t': 'Radiolab', 'tp': 'Science', 'l': 'English'}]},
    {'id': 110, 'p': [{'t': 'Hidden Brain', 'tp': 'Society', 'l': 'English'}, {'t': 'The Art History Babes', 'tp': 'Art', 'l': 'English'}, {'t': 'Waveform', 'tp': 'Technology', 'l': 'English'}, {'t': 'The Bill Simmons Podcast', 'tp': 'Sports', 'l': 'English'}, {'t': 'All Songs Considered', 'tp': 'Music', 'l': 'English'}]},
    {'id': 111, 'p': [{'t': 'Pardon My Take', 'tp': 'Sports', 'l': 'English'}, {'t': 'The Vergecast', 'tp': 'Technology', 'l': 'English'}, {'t': 'My Dad Wrote A Porno', 'tp': 'Comedy', 'l': 'English'}, {'t': 'The Tim Ferriss Show', 'tp': 'Business', 'l': 'English'}]},
    {'id': 112, 'p': [{'t': 'Reflektor', 'tp': 'Music', 'l': 'German'}, {'t': 'Cosmo – Machiavelli', 'tp': 'Society', 'l': 'German'}, {'t': 'Gemischtes Hack', 'tp': 'Comedy', 'l': 'German'}]},
    {'id': 113, 'p': [{'t': 'Einmal täglich Glück', 'tp': 'Health', 'l': 'German'}, {'t': 'Technik und mehr', 'tp': 'Technology', 'l': 'German'}, {'t': 'Handelsblatt Today', 'tp': 'Business', 'l': 'German'}, {'t': 'Kicker meets DAZN', 'tp': 'Sports', 'l': 'German'}, {'t': 'Laut gegen Nazis', 'tp': 'Music', 'l': 'German'}]},
    {'id': 114, 'p': [{'t': 'Laut gegen Nazis', 'tp': 'Music', 'l': 'German'}, {'t': 'Verbrechen', 'tp': 'True Crime', 'l': 'German'}, {'t': 'Herzzeit', 'tp': 'Spirituality', 'l': 'German'}, {'t': 'Handelsblatt Today', 'tp': 'Business', 'l': 'German'}]},
    {'id': 115, 'p': [{'t': 'brand eins Podcast', 'tp': 'Business', 'l': 'German'}, {'t': 'Kicker meets DAZN', 'tp': 'Sports', 'l': 'German'}, {'t': 'Der kreative Flow', 'tp': 'Art', 'l': 'German'}]},
    {'id': 116, 'p': [{'t': 'Deutschland3000', 'tp': 'Society', 'l': 'German'}, {'t': 'Lage der Liga', 'tp': 'Sports', 'l': 'German'}, {'t': 'Gemischtes Hack', 'tp': 'Comedy', 'l': 'German'}]},
    {'id': 117, 'p': [{'t': 'Comedy Bang Bang', 'tp': 'Comedy', 'l': 'English'}, {'t': 'Switched on Pop', 'tp': 'Music', 'l': 'English'}, {'t': 'Hidden Brain', 'tp': 'Society', 'l': 'English'}]},
    {'id': 118, 'p': [{'t': 'On Being', 'tp': 'Spirituality', 'l': 'English'}, {'t': 'The Ezra Klein Show', 'tp': 'Society', 'l': 'English'}, {'t': 'Switched on Pop', 'tp': 'Music', 'l': 'English'}, {'t': 'Comedy Bang Bang', 'tp': 'Comedy', 'l': 'English'}, {'t': 'Radiolab', 'tp': 'Science', 'l': 'English'}]},
    {'id': 119, 'p': [{'t': 'Laut gegen Nazis', 'tp': 'Music', 'l': 'German'}, {'t': 'Apotheken Umschau', 'tp': 'Health', 'l': 'German'}, {'t': 'Gemischtes Hack', 'tp': 'Comedy', 'l': 'German'}]},
    {'id': 120, 'p': [{'t': 'The Vergecast', 'tp': 'Technology', 'l': 'English'}, {'t': 'Radiolab', 'tp': 'Science', 'l': 'English'}, {'t': 'Pardon My Take', 'tp': 'Sports', 'l': 'English'}, {'t': 'Crime Junkie', 'tp': 'True Crime', 'l': 'English'}, {'t': "The Doctor's Farmacy", 'tp': 'Health', 'l': 'English'}]},
    {'id': 121, 'p': [{'t': 'Feel Better Live More', 'tp': 'Health', 'l': 'English'}, {'t': 'Crime Junkie', 'tp': 'True Crime', 'l': 'English'}]},
    {'id': 122, 'p': [{'t': 'Hidden Brain', 'tp': 'Society', 'l': 'English'}, {'t': 'How I Built This', 'tp': 'Business', 'l': 'English'}, {'t': 'Science Vs', 'tp': 'Science', 'l': 'English'}, {'t': 'The Bill Simmons Podcast', 'tp': 'Sports', 'l': 'English'}]},
    {'id': 123, 'p': [{'t': 'On Being', 'tp': 'Spirituality', 'l': 'English'}, {'t': 'Morbid', 'tp': 'True Crime', 'l': 'English'}]},
    {'id': 124, 'p': [{'t': 'Fest & Flauschig', 'tp': 'Comedy', 'l': 'German'}, {'t': 'Die sogenannte Gegenwart', 'tp': 'Society', 'l': 'German'}]},
    {'id': 125, 'p': [{'t': 'Business Wars', 'tp': 'Business', 'l': 'English'}, {'t': 'Song Exploder', 'tp': 'Music', 'l': 'English'}]},
    {'id': 126, 'p': [{'t': 'Apotheken Umschau', 'tp': 'Health', 'l': 'German'}, {'t': 'Buddhaweisheit', 'tp': 'Spirituality', 'l': 'German'}, {'t': 'Baywatch Berlin', 'tp': 'Comedy', 'l': 'German'}, {'t': 'Kicker meets DAZN', 'tp': 'Sports', 'l': 'German'}, {'t': 'Technik und mehr', 'tp': 'Technology', 'l': 'German'}]},
    {'id': 127, 'p': [{'t': 'KUNST PAUSE', 'tp': 'Art', 'l': 'German'}, {'t': 'Deutschland3000', 'tp': 'Society', 'l': 'German'}, {'t': 'Mordlust', 'tp': 'True Crime', 'l': 'German'}, {'t': 'Der spirituelle Podcast', 'tp': 'Spirituality', 'l': 'German'}]},
    {'id': 128, 'p': [{'t': 'Science Vs', 'tp': 'Science', 'l': 'English'}, {'t': 'Song Exploder', 'tp': 'Music', 'l': 'English'}]},
    {'id': 129, 'p': [{'t': 'Business Wars', 'tp': 'Business', 'l': 'English'}, {'t': 'Waveform', 'tp': 'Technology', 'l': 'English'}, {'t': 'The Infinite Monkey Cage', 'tp': 'Science', 'l': 'English'}]},
    {'id': 130, 'p': [{'t': 'brand eins Podcast', 'tp': 'Business', 'l': 'German'}, {'t': 'IQ Wissenschaft und Forschung', 'tp': 'Science', 'l': 'German'}, {'t': 'Die Zeichen des Bösen', 'tp': 'True Crime', 'l': 'German'}, {'t': 'KUNST PAUSE', 'tp': 'Art', 'l': 'German'}]},
    {'id': 131, 'p': [{'t': 'Science Vs', 'tp': 'Science', 'l': 'English'}, {'t': 'Comedy Bang Bang', 'tp': 'Comedy', 'l': 'English'}, {'t': 'Song Exploder', 'tp': 'Music', 'l': 'English'}]},
    {'id': 132, 'p': [{'t': 'All Songs Considered', 'tp': 'Music', 'l': 'English'}, {'t': 'Talk Art', 'tp': 'Art', 'l': 'English'}, {'t': 'The Infinite Monkey Cage', 'tp': 'Science', 'l': 'English'}, {'t': 'Serial', 'tp': 'True Crime', 'l': 'English'}]},
    {'id': 133, 'p': [{'t': 'KUNST PAUSE', 'tp': 'Art', 'l': 'German'}, {'t': 'Mordlust', 'tp': 'True Crime', 'l': 'German'}]},
    {'id': 134, 'p': [{'t': 'Buddhaweisheit', 'tp': 'Spirituality', 'l': 'German'}, {'t': 'Der kreative Flow', 'tp': 'Art', 'l': 'German'}, {'t': 'Lage der Liga', 'tp': 'Sports', 'l': 'German'}, {'t': 'Cosmo – Machiavelli', 'tp': 'Society', 'l': 'German'}, {'t': 'Stern Crime', 'tp': 'True Crime', 'l': 'German'}]},
    {'id': 135, 'p': [{'t': 'Baywatch Berlin', 'tp': 'Comedy', 'l': 'German'}, {'t': 'Logbuch:Netzpolitik', 'tp': 'Technology', 'l': 'German'}, {'t': 'Popcast', 'tp': 'Music', 'l': 'German'}, {'t': 'Handelsblatt Today', 'tp': 'Business', 'l': 'German'}]},
    {'id': 136, 'p': [{'t': 'Feel Better Live More', 'tp': 'Health', 'l': 'English'}, {'t': 'Reply All', 'tp': 'Technology', 'l': 'English'}]},
    {'id': 137, 'p': [{'t': 'How I Built This', 'tp': 'Business', 'l': 'English'}, {'t': 'Song Exploder', 'tp': 'Music', 'l': 'English'}, {'t': 'Crime Junkie', 'tp': 'True Crime', 'l': 'English'}, {'t': 'Science Vs', 'tp': 'Science', 'l': 'English'}]},
    {'id': 138, 'p': [{'t': 'My Favorite Murder', 'tp': 'True Crime', 'l': 'English'}, {'t': 'Code Switch', 'tp': 'Society', 'l': 'English'}, {'t': 'Feel Better Live More', 'tp': 'Health', 'l': 'English'}, {'t': 'ArtCurious', 'tp': 'Art', 'l': 'English'}, {'t': 'Switched on Pop', 'tp': 'Music', 'l': 'English'}]},
    {'id': 139, 'p': [{'t': 'Herzzeit', 'tp': 'Spirituality', 'l': 'German'}, {'t': 'Mordlust', 'tp': 'True Crime', 'l': 'German'}, {'t': 'OMR Podcast', 'tp': 'Business', 'l': 'German'}, {'t': 'Und was machst du am Wochenende?', 'tp': 'Sports', 'l': 'German'}]},
    {'id': 140, 'p': [{'t': 'Oprah’s SuperSoul', 'tp': 'Spirituality', 'l': 'English'}, {'t': 'ArtCurious', 'tp': 'Art', 'l': 'English'}]},
    {'id': 141, 'p': [{'t': 'IQ Wissenschaft und Forschung', 'tp': 'Science', 'l': 'German'}, {'t': 'Die sogenannte Gegenwart', 'tp': 'Society', 'l': 'German'}, {'t': 'CT Uplink', 'tp': 'Technology', 'l': 'German'}]},
    {'id': 142, 'p': [{'t': 'Code Switch', 'tp': 'Society', 'l': 'English'}, {'t': 'The Daily Show: Ears Edition', 'tp': 'Comedy', 'l': 'English'}]},
    {'id': 143, 'p': [{'t': 'Business Wars', 'tp': 'Business', 'l': 'English'}, {'t': 'Crime Junkie', 'tp': 'True Crime', 'l': 'English'}, {'t': 'All Songs Considered', 'tp': 'Music', 'l': 'English'}, {'t': 'Feel Better Live More', 'tp': 'Health', 'l': 'English'}]},
    {'id': 144, 'p': [{'t': 'Wissenschaft im Brennpunkt', 'tp': 'Science', 'l': 'German'}, {'t': 'Popcast', 'tp': 'Music', 'l': 'German'}, {'t': 'Buddhaweisheit', 'tp': 'Spirituality', 'l': 'German'}, {'t': 'Verbrechen', 'tp': 'True Crime', 'l': 'German'}, {'t': 'Baywatch Berlin', 'tp': 'Comedy', 'l': 'German'}]},
    {'id': 145, 'p': [{'t': 'Und was machst du am Wochenende?', 'tp': 'Sports', 'l': 'German'}, {'t': 'Gesund Leben', 'tp': 'Health', 'l': 'German'}, {'t': 'Stern Crime', 'tp': 'True Crime', 'l': 'German'}, {'t': 'Popcast', 'tp': 'Music', 'l': 'German'}]},
    {'id': 146, 'p': [{'t': 'Oprah’s SuperSoul', 'tp': 'Spirituality', 'l': 'English'}, {'t': 'The Infinite Monkey Cage', 'tp': 'Science', 'l': 'English'}, {'t': "The Doctor's Farmacy", 'tp': 'Health', 'l': 'English'}]},
    {'id': 147, 'p': [{'t': 'Und was machst du am Wochenende?', 'tp': 'Sports', 'l': 'German'}, {'t': 'CT Uplink', 'tp': 'Technology', 'l': 'German'}, {'t': 'KUNST DRUCK', 'tp': 'Art', 'l': 'German'}, {'t': 'brand eins Podcast', 'tp': 'Business', 'l': 'German'}, {'t': 'Popcast', 'tp': 'Music', 'l': 'German'}]},
    {'id': 148, 'p': [{'t': 'Reply All', 'tp': 'Technology', 'l': 'English'}, {'t': 'The Infinite Monkey Cage', 'tp': 'Science', 'l': 'English'}]},
    {'id': 149, 'p': [{'t': 'Buddhaweisheit', 'tp': 'Spirituality', 'l': 'German'}, {'t': 'Verbrechen', 'tp': 'True Crime', 'l': 'German'}, {'t': 'Einmal täglich Glück', 'tp': 'Health', 'l': 'German'}, {'t': 'Reflektor', 'tp': 'Music', 'l': 'German'}, {'t': 'Der kreative Flow', 'tp': 'Art', 'l': 'German'}]},
    {'id': 150, 'p': [{'t': 'Apotheken Umschau', 'tp': 'Health', 'l': 'German'}, {'t': 'Herzzeit', 'tp': 'Spirituality', 'l': 'German'}, {'t': 'OMR Podcast', 'tp': 'Business', 'l': 'German'}]},
    {'id': 151, 'p': [{'t': 'All Songs Considered', 'tp': 'Music', 'l': 'English'}, {'t': 'The Daily Show: Ears Edition', 'tp': 'Comedy', 'l': 'English'}, {'t': 'Morbid', 'tp': 'True Crime', 'l': 'English'}, {'t': 'Pardon My Take', 'tp': 'Sports', 'l': 'English'}]},
    {'id': 152, 'p': [{'t': 'Code Switch', 'tp': 'Society', 'l': 'English'}, {'t': 'On Being', 'tp': 'Spirituality', 'l': 'English'}]},
    {'id': 153, 'p': [{'t': 'The Tim Ferriss Show', 'tp': 'Business', 'l': 'English'}, {'t': 'Code Switch', 'tp': 'Society', 'l': 'English'}, {'t': 'Science Vs', 'tp': 'Science', 'l': 'English'}, {'t': 'Comedy Bang Bang', 'tp': 'Comedy', 'l': 'English'}]},
    {'id': 154, 'p': [{'t': 'Serial', 'tp': 'True Crime', 'l': 'English'}, {'t': 'The Tim Ferriss Show', 'tp': 'Business', 'l': 'English'}, {'t': 'On Being', 'tp': 'Spirituality', 'l': 'English'}]},
    {'id': 155, 'p': [{'t': 'Kicker meets DAZN', 'tp': 'Sports', 'l': 'German'}, {'t': 'Gesund Leben', 'tp': 'Health', 'l': 'German'}]},
    {'id': 156, 'p': [{'t': 'Baywatch Berlin', 'tp': 'Comedy', 'l': 'German'}, {'t': 'Mordlust', 'tp': 'True Crime', 'l': 'German'}]},
    {'id': 157, 'p': [{'t': 'The Bill Simmons Podcast', 'tp': 'Sports', 'l': 'English'}, {'t': 'How I Built This', 'tp': 'Business', 'l': 'English'}, {'t': 'My Dad Wrote A Porno', 'tp': 'Comedy', 'l': 'English'}, {'t': 'Healthier Together', 'tp': 'Health', 'l': 'English'}]},
    {'id': 158, 'p': [{'t': 'The Art History Babes', 'tp': 'Art', 'l': 'English'}, {'t': 'Oprah’s SuperSoul', 'tp': 'Spirituality', 'l': 'English'}, {'t': "The Doctor's Farmacy", 'tp': 'Health', 'l': 'English'}]},
    {'id': 159, 'p': [{'t': "The Doctor's Farmacy", 'tp': 'Health', 'l': 'English'}, {'t': 'Pardon My Take', 'tp': 'Sports', 'l': 'English'}]},
    {'id': 160, 'p': [{'t': 'Waveform', 'tp': 'Technology', 'l': 'English'}, {'t': 'Comedy Bang Bang', 'tp': 'Comedy', 'l': 'English'}, {'t': 'Song Exploder', 'tp': 'Music', 'l': 'English'}]},
    {'id': 161, 'p': [{'t': 'Einmal täglich Glück', 'tp': 'Health', 'l': 'German'}, {'t': 'Herzzeit', 'tp': 'Spirituality', 'l': 'German'}, {'t': 'Mordlust', 'tp': 'True Crime', 'l': 'German'}, {'t': 'Die sogenannte Gegenwart', 'tp': 'Society', 'l': 'German'}, {'t': 'Und was machst du am Wochenende?', 'tp': 'Sports', 'l': 'German'}]},
    {'id': 162, 'p': [{'t': 'Gesund Leben', 'tp': 'Health', 'l': 'German'}, {'t': 'Der kreative Flow', 'tp': 'Art', 'l': 'German'}, {'t': 'Die Zeichen des Bösen', 'tp': 'True Crime', 'l': 'German'}, {'t': 'Reflektor', 'tp': 'Music', 'l': 'German'}, {'t': 'Buddhaweisheit', 'tp': 'Spirituality', 'l': 'German'}]},
    {'id': 163, 'p': [{'t': 'The Tim Ferriss Show', 'tp': 'Business', 'l': 'English'}, {'t': 'The Ezra Klein Show', 'tp': 'Society', 'l': 'English'}, {'t': 'On Being', 'tp': 'Spirituality', 'l': 'English'}]},
    {'id': 164, 'p': [{'t': 'Song Exploder', 'tp': 'Music', 'l': 'English'}, {'t': 'Healthier Together', 'tp': 'Health', 'l': 'English'}, {'t': 'Comedy Bang Bang', 'tp': 'Comedy', 'l': 'English'}, {'t': 'Waveform', 'tp': 'Technology', 'l': 'English'}]},
    {'id': 165, 'p': [{'t': 'Waveform', 'tp': 'Technology', 'l': 'English'}, {'t': 'The Daily Show: Ears Edition', 'tp': 'Comedy', 'l': 'English'}, {'t': 'Feel Better Live More', 'tp': 'Health', 'l': 'English'}, {'t': 'ArtCurious', 'tp': 'Art', 'l': 'English'}]},
    {'id': 166, 'p': [{'t': 'The Ezra Klein Show', 'tp': 'Society', 'l': 'English'}, {'t': "The Doctor's Farmacy", 'tp': 'Health', 'l': 'English'}, {'t': 'The Vergecast', 'tp': 'Technology', 'l': 'English'}, {'t': 'My Dad Wrote A Porno', 'tp': 'Comedy', 'l': 'English'}]},
    {'id': 167, 'p': [{'t': 'The Infinite Monkey Cage', 'tp': 'Science', 'l': 'English'}, {'t': 'Code Switch', 'tp': 'Society', 'l': 'English'}]},
    {'id': 168, 'p': [{'t': 'Comedy Bang Bang', 'tp': 'Comedy', 'l': 'English'}, {'t': 'Song Exploder', 'tp': 'Music', 'l': 'English'}, {'t': 'Pardon My Take', 'tp': 'Sports', 'l': 'English'}]},
    {'id': 169, 'p': [{'t': 'How I Built This', 'tp': 'Business', 'l': 'English'}, {'t': 'Football Weekly', 'tp': 'Sports', 'l': 'English'}, {'t': 'Oprah’s SuperSoul', 'tp': 'Spirituality', 'l': 'English'}, {'t': 'Reply All', 'tp': 'Technology', 'l': 'English'}]},
    {'id': 170, 'p': [{'t': 'Feel Better Live More', 'tp': 'Health', 'l': 'English'}, {'t': 'The Art History Babes', 'tp': 'Art', 'l': 'English'}, {'t': 'Switched on Pop', 'tp': 'Music', 'l': 'English'}]},
    {'id': 171, 'p': [{'t': 'Deutschland3000', 'tp': 'Society', 'l': 'German'}, {'t': 'Und was machst du am Wochenende?', 'tp': 'Sports', 'l': 'German'}, {'t': 'Laut gegen Nazis', 'tp': 'Music', 'l': 'German'}, {'t': 'Der spirituelle Podcast', 'tp': 'Spirituality', 'l': 'German'}, {'t': 'CT Uplink', 'tp': 'Technology', 'l': 'German'}]},
    {'id': 172, 'p': [{'t': 'All Songs Considered', 'tp': 'Music', 'l': 'English'}, {'t': 'On Being', 'tp': 'Spirituality', 'l': 'English'}, {'t': 'ArtCurious', 'tp': 'Art', 'l': 'English'}]},
    {'id': 173, 'p': [{'t': 'Oprah’s SuperSoul', 'tp': 'Spirituality', 'l': 'English'}, {'t': 'Healthier Together', 'tp': 'Health', 'l': 'English'}, {'t': 'Science Vs', 'tp': 'Science', 'l': 'English'}, {'t': 'The Bill Simmons Podcast', 'tp': 'Sports', 'l': 'English'}]},
    {'id': 174, 'p': [{'t': 'My Dad Wrote A Porno', 'tp': 'Comedy', 'l': 'English'}, {'t': 'Talk Art', 'tp': 'Art', 'l': 'English'}, {'t': 'The Tim Ferriss Show', 'tp': 'Business', 'l': 'English'}]},
    {'id': 175, 'p': [{'t': 'Lage der Liga', 'tp': 'Sports', 'l': 'German'}, {'t': 'Verbrechen', 'tp': 'True Crime', 'l': 'German'}, {'t': 'Cosmo – Machiavelli', 'tp': 'Society', 'l': 'German'}, {'t': 'Einmal täglich Glück', 'tp': 'Health', 'l': 'German'}, {'t': 'OMR Podcast', 'tp': 'Business', 'l': 'German'}]},
    {'id': 176, 'p': [{'t': 'Hidden Brain', 'tp': 'Society', 'l': 'English'}, {'t': 'Talk Art', 'tp': 'Art', 'l': 'English'}]},
    {'id': 177, 'p': [{'t': 'Hidden Brain', 'tp': 'Society', 'l': 'English'}, {'t': "The Doctor's Farmacy", 'tp': 'Health', 'l': 'English'}, {'t': 'The Vergecast', 'tp': 'Technology', 'l': 'English'}]},
    {'id': 178, 'p': [{'t': 'On Being', 'tp': 'Spirituality', 'l': 'English'}, {'t': 'Radiolab', 'tp': 'Science', 'l': 'English'}, {'t': 'The Daily Show: Ears Edition', 'tp': 'Comedy', 'l': 'English'}, {'t': 'My Favorite Murder', 'tp': 'True Crime', 'l': 'English'}]},
    {'id': 179, 'p': [{'t': 'A New Earth', 'tp': 'Spirituality', 'l': 'English'}, {'t': 'How I Built This', 'tp': 'Business', 'l': 'English'}]},
]

# 2. DataFrame erstellen
# Jede Zeile: user_id + Liste aller gehörten Titel
df = pd.DataFrame([
    {"user_id": u["id"], "titles": [p["t"] for p in u["p"]]}
    for u in user_podcasts
])

# 3. One-Hot-Encoding (MultiLabelBinarizer)
mlb = MultiLabelBinarizer(sparse_output=True)
X_titles = mlb.fit_transform(df["titles"])
titles = mlb.classes_

# 4. Positive (+) und Negative (-) Beispiele generieren
X_list, y_list = [], []
for idx, row in df.iterrows():
    user_vec = X_titles[idx]           # sparse user-feature vector
    heard = row["titles"]            # echte gehört-Podcasts
    # + Beispiele
    for t in heard:
        t_idx = np.where(titles == t)[0][0]
        X_list.append(user_vec.toarray().ravel())
        y_list.append(t_idx)
    # - Beispiele: gleiche Anzahl zufällig aus dem Rest
    candidates = list(set(titles) - set(heard))
    neg = np.random.choice(candidates, size=len(heard), replace=False)
    for t in neg:
        t_idx = np.where(titles == t)[0][0]
        X_list.append(user_vec.toarray().ravel())
        y_list.append(t_idx)

# In Arrays umwandeln
X = np.vstack(X_list)
y = np.array(y_list)

# 5. Train/Test-Split
gbt = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_test, y_train, y_test = gbt

# 6. Modell trainieren
clf = LogisticRegression(multi_class='multinomial', solver='saga', max_iter=200, random_state=42)
clf.fit(X_train, y_train)

# 7. Auswertung
y_pred = clf.predict(X_test)                                 # <<– hier einfügen

# Accuracy
print(f"Accuracy: {accuracy_score(y_test, y_pred):.3f}")

# Top-5 Accuracy (mit allen Klassen)
all_labels = np.arange(len(titles))
top5 = top_k_accuracy_score(
    y_test,
    clf.predict_proba(X_test),
    k=5,
    labels=all_labels
)
print(f"Top-5 Accuracy: {top5:.3f}")



# 8. Empfehlungsfunktion

def recommend_for_user(user_id, k=5):
    i = df.index[df['user_id'] == user_id][0]
    vec = X_titles[i]
    probs = clf.predict_proba(vec)[0]
    top = np.argsort(probs)[::-1][:k]
    return [(titles[j], float(probs[j])) for j in top]

# Beispiel: Empfehlungen für user_id 0
print("Empfehlungen für Nutzer 0:", recommend_for_user(0))

import joblib

# Modell und Encoder auf der Festplatte speichern:
joblib.dump(clf, "clf.joblib")
joblib.dump(mlb, "mlb.joblib")

print("Modell und Encoder wurden gespeichert.")
