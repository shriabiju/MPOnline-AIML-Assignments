"""
train_model.py
---------------
Downloads the Sentiment140 dataset, cleans it, trains the three models used
by the chatbot (sentiment ensemble, intent classifier, conversational
retrieval model), and saves everything to chatbot_models/.

Run this ONCE locally before running the Streamlit app:
    python train_model.py

This will create a chatbot_models/ folder with the .pkl files that app.py
loads. Commit that folder to your repo so Render doesn't need to retrain
on every deploy.
"""

import os
import re
import random
import zipfile
import urllib.request

import numpy as np
import pandas as pd
import joblib

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import VotingClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from sklearn.metrics.pairwise import cosine_similarity

random.seed(42)
np.random.seed(42)

print("downloading NLTK data...")
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)
nltk.download('averaged_perceptron_tagger_eng', quiet=True)


# ---------------------------------------------------------------------------
# 1. Download / load the Sentiment140 dataset (1.6M tweets)
# ---------------------------------------------------------------------------
def load_dataset():
    csv_file = "sentiment140.csv"

    if not os.path.exists(csv_file):
        print("downloading sentiment140 (1.6M tweets, ~80MB)...")
        url = "https://cs.stanford.edu/people/alecmgo/trainingandtestdata.zip"
        zip_file = "sentiment140.zip"

        try:
            urllib.request.urlretrieve(url, zip_file)
        except Exception:
            # ssl workaround (some machines / mac)
            import ssl
            ctx = ssl.create_default_context()
            ctx.check_hostname = False
            ctx.verify_mode = ssl.CERT_NONE
            opener = urllib.request.build_opener(urllib.request.HTTPSHandler(context=ctx))
            urllib.request.install_opener(opener)
            urllib.request.urlretrieve(url, zip_file)

        with zipfile.ZipFile(zip_file, 'r') as z:
            z.extractall(".")

        os.rename("training.1600000.processed.noemoticon.csv", csv_file)
        if os.path.exists(zip_file):
            os.remove(zip_file)
        if os.path.exists("testdata.manual.2009.06.14.csv"):
            os.remove("testdata.manual.2009.06.14.csv")
        print("download complete")
    else:
        print("dataset already exists, skipping download")

    df_full = pd.read_csv(csv_file, encoding='latin-1', header=None,
                           names=['sentiment', 'id', 'date', 'query', 'user', 'text'])
    df_full['sentiment'] = df_full['sentiment'].map({0: 0, 4: 1})

    print(f"loaded {len(df_full):,} tweets")
    print(f"positive: {(df_full['sentiment'] == 1).sum():,}")
    print(f"negative: {(df_full['sentiment'] == 0).sum():,}")
    return df_full


# ---------------------------------------------------------------------------
# 2. Preprocessing with negation handling
# ---------------------------------------------------------------------------
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

negation = {"not", "no", "nor", "never", "neither", "nobody", "nothing",
            "nowhere", "cannot", "cant", "dont", "didnt", "doesnt",
            "isnt", "wasnt", "werent", "wont", "wouldnt", "shouldnt",
            "couldnt", "aint", "hardly", "barely"}

stop_words_clean = stop_words - negation


def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'#', '', text)
    text = re.sub(r'\brt\b', '', text)
    text = re.sub(r"n't", " not", text)
    text = re.sub(r"can't", "can not", text)
    text = re.sub(r"won't", "will not", text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    tokens = word_tokenize(text)
    tokens = [lemmatizer.lemmatize(w) for w in tokens if w not in stop_words_clean and len(w) > 1]

    result = []
    negate = False
    for t in tokens:
        if t in negation:
            negate = True
            result.append(t)
        elif negate:
            result.append(f"NOT_{t}")
            negate = False
        else:
            result.append(t)
    return ' '.join(result)


def clean_text_light(text):
    text = str(text).lower().strip()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


# ---------------------------------------------------------------------------
# 3. Intent training data (6 classes) + conversational retrieval pairs
# ---------------------------------------------------------------------------
INTENT_DATA = {
    'text': [
        "hello", "hi there", "hey", "good morning", "good afternoon", "good evening",
        "howdy", "hey whats up", "hi how are you", "hello there", "greetings",
        "hey hey", "hiya", "morning", "evening", "yo", "sup", "hey buddy",
        "hi everyone", "hello friend", "hey there how are you", "good day",
        "whats good", "hey how is it going", "hello hello", "hi", "hi hi",

        "bye", "goodbye", "see you later", "take care", "gotta go", "see ya",
        "bye bye", "later", "have a good day", "catch you later", "peace out",
        "im leaving", "talk to you later", "good night", "signing off",
        "i have to go now", "until next time", "farewell", "see you soon",
        "bye for now", "im heading out", "thats all for now", "time to go",
        "alright bye", "ok goodbye",

        "what can you do", "how does this work", "what are your features",
        "tell me about yourself", "what is sentiment analysis", "how do you analyze text",
        "explain what you do", "what is nlp", "how were you trained",
        "what dataset did you use", "what is polarity", "what is subjectivity",
        "how accurate are you", "can you explain that", "what does that mean",
        "how do you detect mood", "what model do you use", "tell me more",
        "what is machine learning", "how does the analysis work",
        "what is tfidf", "explain your features", "what are you",
        "who made you", "what is your purpose",

        "i feel so happy today", "im really sad right now", "i am angry",
        "feeling great", "im depressed", "i feel amazing", "im anxious",
        "i feel lonely", "so excited about this", "im frustrated",
        "feeling blessed", "i feel terrible", "im in a good mood",
        "i feel stressed", "im so proud of myself", "feeling hopeless",
        "i love my life", "everything is going wrong", "im grateful",
        "i feel overwhelmed", "today was the best day", "i feel empty",
        "im scared", "feeling confident", "i hate everything",

        "i think this is great", "in my opinion this is bad", "i believe we should",
        "the movie was amazing", "that restaurant was terrible", "i love this song",
        "this product is awful", "best experience ever", "worst day of my life",
        "i really enjoyed the book", "the service was horrible", "that was incredible",
        "i dont like this at all", "this is my favorite", "its not worth it",
        "totally recommend this", "i disagree with that", "this is overrated",
        "that was disappointing", "absolutely loved it", "its the worst",
        "pretty good actually", "not bad at all", "could be better",
        "i strongly support this",

        "analyze this text for me", "what is the sentiment of this",
        "check the mood of this sentence", "rate this text", "is this positive or negative",
        "tell me the sentiment", "analyze my message", "what emotion is this",
        "detect the sentiment", "how does this sound", "is this happy or sad",
        "score this text", "evaluate this statement", "read the mood here",
        "what vibe does this give", "break down this text", "sentiment check",
        "analyze the following", "how positive is this", "classify this text",
        "whats the polarity of this", "run analysis on this", "check this",
        "is this negative", "what do you read from this",
    ],
    'intent': (
        ['greeting'] * 27 + ['farewell'] * 25 + ['question'] * 25 +
        ['emotion'] * 25 + ['opinion'] * 25 + ['analysis'] * 25
    )
}

CONV_PAIRS = [
    ("hi", "Hello!"), ("hello", "Hi!"), ("hey", "Hey!"),
    ("hi there", "Hello! How's it going?"), ("hello there", "Hi! Nice to see you."),
    ("hey there", "Hey! What's up?"), ("good morning", "Good morning!"),
    ("good afternoon", "Good afternoon!"), ("good evening", "Good evening!"),
    ("howdy", "Howdy!"), ("yo", "Yo!"), ("sup", "Hey! What's good?"),
    ("hiya", "Hiya!"), ("greetings", "Greetings!"), ("hey buddy", "Hey!"),
    ("hi everyone", "Hello!"), ("hey hey", "Hey!"), ("hi hi", "Hello hello!"),

    ("bye", "Bye!"), ("goodbye", "Goodbye!"), ("bye bye", "Bye bye!"),
    ("see you later", "See you later!"), ("see ya", "See ya!"),
    ("take care", "Take care!"), ("good night", "Good night!"),
    ("gotta go", "Alright, see you!"), ("later", "Later!"),
    ("farewell", "Farewell!"), ("see you soon", "See you soon!"),
    ("peace out", "Peace!"), ("im leaving", "Catch you later!"),

    ("how are you", "I'm doing well, thanks! How about you?"),
    ("how are you doing", "Doing great! What about you?"),
    ("how is it going", "Going well! You?"),
    ("whats up", "Not much! What about you?"),
    ("how do you feel", "Running smoothly! How are you?"),
    ("hows your day", "My day is good! How about yours?"),

    ("what can you do", "I analyze sentiment using an ML ensemble trained on 1.6M tweets, "
     "detect intent, track mood, and chat. Try saying something!"),
    ("how does this work", "I use an ensemble of Logistic Regression, SVM, and Naive Bayes "
     "with TF-IDF for sentiment, plus cosine similarity for conversation."),
    ("what are you", "I'm an NLP chatbot trained on the Sentiment140 dataset."),
    ("who made you", "I was built as an NLP project using scikit-learn, NLTK, and TextBlob."),
    ("what is sentiment analysis", "It's an NLP technique to determine if text is positive or negative."),
    ("what is nlp", "NLP is Natural Language Processing — helping computers understand human language."),
    ("help", "I analyze sentiment, detect intent, extract keywords, and track mood. Just type anything!"),

    ("thank you", "You're welcome!"), ("thanks", "No problem!"),
    ("thanks a lot", "Glad I could help!"), ("appreciate it", "Anytime!"),

    ("tell me a joke", "Why do programmers prefer dark mode? Because light attracts bugs!"),
    ("make me laugh", "What do you call a fake noodle? An impasta!"),
]


def main():
    df_full = load_dataset()

    # balanced 100k sample
    df_neg = df_full[df_full['sentiment'] == 0].sample(50000, random_state=42)
    df_pos = df_full[df_full['sentiment'] == 1].sample(50000, random_state=42)
    df = pd.concat([df_neg, df_pos]).reset_index(drop=True)

    print(f"sampled {len(df)} tweets for training")
    print("cleaning text (this takes a couple minutes)...")
    df['clean_text'] = df['text'].apply(clean_text)
    df = df[df['clean_text'].str.strip() != ''].reset_index(drop=True)
    print(f"done. {len(df)} tweets after cleaning")

    # --- 1. sentiment classifier (ensemble of 3 models) ---
    X = df['clean_text']
    y = df['sentiment']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    tfidf_sentiment = TfidfVectorizer(
        max_features=100000, ngram_range=(1, 3), min_df=2, sublinear_tf=True)
    X_train_tfidf = tfidf_sentiment.fit_transform(X_train)
    X_test_tfidf = tfidf_sentiment.transform(X_test)

    lr = LogisticRegression(max_iter=1000, C=1.0, random_state=42)
    nb = MultinomialNB(alpha=0.1)
    svm = CalibratedClassifierCV(LinearSVC(max_iter=2000, C=1.0, random_state=42))

    sentiment_model = VotingClassifier(
        estimators=[('lr', lr), ('nb', nb), ('svm', svm)], voting='soft')

    print("training ensemble (LogReg + SVM + NaiveBayes)... this takes a minute")
    sentiment_model.fit(X_train_tfidf, y_train)

    y_pred = sentiment_model.predict(X_test_tfidf)
    print(f"ensemble accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")
    for name, model in sentiment_model.named_estimators_.items():
        pred = model.predict(X_test_tfidf)
        print(f"  {name} alone: {accuracy_score(y_test, pred) * 100:.2f}%")

    # --- 2. intent classifier ---
    intent_df = pd.DataFrame(INTENT_DATA)
    intent_df['clean'] = intent_df['text'].apply(clean_text_light)

    intent_pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=5000, ngram_range=(1, 2))),
        ('clf', MultinomialNB(alpha=0.1))
    ])
    intent_pipeline.fit(intent_df['clean'], intent_df['intent'])
    print("intent classifier trained")

    # --- 3. conversational retrieval model ---
    conv_patterns_clean = [clean_text_light(p) for p, r in CONV_PAIRS]
    conv_responses = [r for p, r in CONV_PAIRS]

    conv_tfidf = TfidfVectorizer(ngram_range=(1, 2))
    conv_vectors = conv_tfidf.fit_transform(conv_patterns_clean)
    print("conversational model trained")

    # --- save everything ---
    model_dir = "chatbot_models"
    os.makedirs(model_dir, exist_ok=True)

    joblib.dump(sentiment_model, f"{model_dir}/sentiment_model.pkl")
    joblib.dump(tfidf_sentiment, f"{model_dir}/tfidf_sentiment.pkl")
    joblib.dump(intent_pipeline, f"{model_dir}/intent_pipeline.pkl")
    joblib.dump(conv_tfidf, f"{model_dir}/conv_tfidf.pkl")
    joblib.dump(conv_vectors, f"{model_dir}/conv_vectors.pkl")
    joblib.dump(conv_responses, f"{model_dir}/conv_responses.pkl")

    print(f"\nall models saved to {model_dir}/")
    print(os.listdir(model_dir))


if __name__ == "__main__":
    main()
