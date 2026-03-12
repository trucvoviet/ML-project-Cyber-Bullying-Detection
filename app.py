import streamlit as st
import joblib
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from bs4 import BeautifulSoup
import contractions

# Download required nltk data
nltk.download("stopwords")
nltk.download("wordnet")


# Load saved model + vectorizer

model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")


# Build stopword set

stop = set(stopwords.words("english"))

negation_words = {
    "not",
    "no",
    "never",
    "neither",
    "nor",
    "nothing",
    "nowhere",
    "nobody",
    "none",
    "cannot",
    "without",
    "against",
    "hardly",
    "scarcely",
    "barely",
    "doesnt",
    "isnt",
    "wasnt",
    "shouldnt",
    "wouldnt",
    "couldnt",
    "wont",
    "cant",
    "dont",
    "didnt",
    "hadnt",
    "hasnt",
    "havent",
    "neednt",
    "mightnt",
    "mustnt",
}

stop = stop - negation_words


# Text preprocessing functions


def expand_contractions(text):
    return contractions.fix(text)


def negate_sequence(text):

    negation_tokens = {
        "not",
        "no",
        "never",
        "nobody",
        "nothing",
        "nowhere",
        "neither",
        "nor",
        "cannot",
        "without",
        "hardly",
        "scarcely",
        "barely",
    }

    clause_breakers = {"but", "however", "although", "though", "yet"}

    tokens = text.split()
    result = []
    negating = False

    for token in tokens:

        clean_token = token.rstrip(".,!?;:")

        if clean_token in negation_tokens:
            negating = True
            result.append(token)

        elif clean_token in clause_breakers or token.endswith((".", "!", "?", ";")):
            negating = False
            result.append(token)

        elif negating and clean_token not in stop:
            result.append("NOT_" + clean_token)

        else:
            result.append(token)

    return " ".join(result)


def preprocess_text(text):

    wl = WordNetLemmatizer()

    text = BeautifulSoup(text, "html.parser").get_text()

    text = expand_contractions(text)

    emoji_clean = re.compile(
        "["
        "\U0001f600-\U0001f64f"
        "\U0001f300-\U0001f5ff"
        "\U0001f680-\U0001f6ff"
        "\U0001f1e0-\U0001f1ff"
        "\U00002702-\U000027b0"
        "\U000024c2-\U0001f251"
        "]+",
        flags=re.UNICODE,
    )

    text = emoji_clean.sub(r"", text)

    text = re.sub(r"http\S+", "", text)

    text = re.sub(r"[^a-zA-Z\s]", "", text.lower())

    text = negate_sequence(text)

    tokens = []

    for word in text.split():

        if word.startswith("NOT_"):
            root = word[4:]
            if root.isalpha():
                tokens.append("NOT_" + wl.lemmatize(root))

        elif word not in stop and word.isalpha():
            tokens.append(wl.lemmatize(word))

    return " ".join(tokens)


# Streamlit UI


st.set_page_config(
    page_title="Cyberbullying Detector", page_icon="🛑", layout="centered"
)

st.title("🛑 Cyberbullying Detection App")
st.write(
    "This application detects **cyberbullying or toxic content** in text using a trained Machine Learning model."
)

st.write("---")

user_input = st.text_area("Enter a sentence to analyze:", height=150)

if st.button("Analyze"):

    if user_input.strip() == "":
        st.warning("Please enter some text.")

    else:

        processed = preprocess_text(user_input)

        X = vectorizer.transform([processed])

        prediction = model.predict(X)[0]

        st.write("### Result")

        if prediction == 1:
            st.error("⚠️ Bullying detected")
        else:
            st.success("✅ Non-bullying content")