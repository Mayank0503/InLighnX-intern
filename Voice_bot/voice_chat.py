import speech_recognition as sr
import pyttsx3
import datetime
import wikipediaapi
import webbrowser
import os
import time
import subprocess
import ctypes
import winsound
import difflib  # For fuzzy matching

# Initialize the text-to-speech engine
engine = pyttsx3.init()

def speak(audio):
    print("Bot:", audio)
    engine.say(audio)
    engine.runAndWait()

def wishMe():
    hour = int(datetime.datetime.now().hour)
    if 0 <= hour < 12:
        speak("Good Morning!")
    elif 12 <= hour < 18:
        speak("Good Afternoon!")
    else:
        speak("Good Evening!")
    speak("I am your assistant. How may I help you?")

def takeCommand():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        print("Listening... Say something now.")
        winsound.Beep(1000, 300)  
        r.pause_threshold = 1
        audio = r.listen(source)
        print("Audio captured, now recognizing...")

    try:
        print("Processing voice input...")
        query = r.recognize_google(audio, language='en-in')
        print(f"User said: {query}")
    except Exception as e:
        print("Could not understand audio. Error:", e)
        return "None"
    return query.lower()

def searchWikipedia(query):
    wiki_wiki = wikipediaapi.Wikipedia('en')
    topic = query.replace("search wikipedia for", "").strip()
    page = wiki_wiki.page(topic)
    if page.exists():
        speak(f"According to Wikipedia: {page.summary[:300]}")
    else:
        speak("Sorry, I couldn't find anything on that topic.")

def openWebsite(query):
    site = query.replace("open", "").strip().replace(" ", "")
    url = f"https://{site}.com"
    try:
        webbrowser.open(url)
        speak(f"Opening {site}")
    except:
        speak("Sorry, I couldn't open that website.")

# Fuzzy matching helper
def is_similar(query, target, threshold=0.75):
    return difflib.SequenceMatcher(None, query, target).ratio() >= threshold

def executeCommand(query):
    if 'open' in query:
        openWebsite(query)
    elif 'time' in query:
        strTime = datetime.datetime.now().strftime("%H:%M:%S")
        speak(f"The time is {strTime}")
    elif 'shutdown' in query:
        speak("Shutting down the system.")
        os.system("shutdown /s /t 1")
    elif 'restart' in query:
        speak("Restarting the system.")
        os.system("shutdown /r /t 1")
    elif 'lock' in query:
        speak("Locking the system.")
        ctypes.windll.user32.LockWorkStation()
    elif 'write a note' in query:
        speak("What should I write?")
        note = takeCommand()
        with open("note.txt", "w") as f:
            f.write(note)
        speak("Note saved.")
    elif 'search wikipedia' in query or 'wikipedia' in query:
        searchWikipedia(query)
    elif 'hello' in query:
        speak("Hello! How can I assist you?")
    elif is_similar(query, 'how are you'):
        speak("I'm functioning as expected!")
    elif 'exit' in query or 'quit' in query:
        speak("Goodbye!")
        exit()
    else:
        speak("Sorry, I didn't understand that.")

if __name__ == "__main__":
    wishMe()
    while True:
        query = takeCommand()
        if query != "None":
            executeCommand(query)
