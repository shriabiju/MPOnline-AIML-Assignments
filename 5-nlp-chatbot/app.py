import streamlit as st
import joblib
import numpy as np
import re
import random
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag
from textblob import TextBlob
from sklearn.metrics.pairwise import cosine_similarity

nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)
nltk.download('averaged_perceptron_tagger_eng', quiet=True)

st.set_page_config(page_title="NLP Sentiment Chatbot", page_icon="💬",
                   layout="centered", initial_sidebar_state="expanded")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
.main { background-color: #0f1117; }
.header-gradient {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
    padding: 30px; border-radius: 16px; margin-bottom: 24px;
    border: 1px solid rgba(255,255,255,0.06);
    box-shadow: 0 4px 24px rgba(0,0,0,0.3);
}
.header-gradient h1 {
    background: linear-gradient(90deg, #a78bfa, #60a5fa, #34d399);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    font-family: 'Inter', sans-serif; font-weight: 700; font-size: 2em;
    margin: 0 0 8px 0;
}
.header-gradient p { color: #94a3b8; font-size: 1em; margin: 0; }
.header-gradient .author {
    color: #60a5fa; font-size: 0.95em; font-weight: 600; margin-top: 10px;
}
.tag {
    display: inline-block; background: rgba(139,92,246,0.12); color: #a78bfa;
    padding: 4px 12px; border-radius: 20px; font-size: 0.78em; margin: 3px 4px;
    border: 1px solid rgba(139,92,246,0.2); font-family: 'Inter', sans-serif;
}
.analysis-card {
    background: #1e1e2e; border-radius: 12px; padding: 16px 20px;
    margin: 8px 0 16px 0; border: 1px solid rgba(255,255,255,0.06);
    font-family: 'Inter', monospace; font-size: 0.85em; color: #cbd5e1; line-height: 1.6;
}
.analysis-card .label { color: #a78bfa; font-weight: 600; }
.pos-tag { color: #34d399; }
.neg-tag { color: #f87171; }
.neu-tag { color: #fbbf24; }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_models():
    d = "chatbot_models"
    return {
        'sentiment_model': joblib.load(f"{d}/sentiment_model.pkl"),
        'tfidf_sentiment': joblib.load(f"{d}/tfidf_sentiment.pkl"),
        'intent_pipeline': joblib.load(f"{d}/intent_pipeline.pkl"),
        'conv_tfidf': joblib.load(f"{d}/conv_tfidf.pkl"),
        'conv_vectors': joblib.load(f"{d}/conv_vectors.pkl"),
        'conv_responses': joblib.load(f"{d}/conv_responses.pkl"),
    }

models = load_models()
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

def analyze_sentiment(text):
    cleaned = clean_text(text)
    vec = models['tfidf_sentiment'].transform([cleaned])
    pred = models['sentiment_model'].predict(vec)[0]
    prob = models['sentiment_model'].predict_proba(vec)[0]
    conf = max(prob) * 100
    if conf < 60:
        label = "Neutral"
    elif pred == 1:
        label = "Positive"
    else:
        label = "Negative"
    blob = TextBlob(text)
    return {'label': label, 'confidence': conf,
            'polarity': blob.sentiment.polarity, 'subjectivity': blob.sentiment.subjectivity}

def classify_intent(text):
    cleaned = clean_text_light(text)
    intent = models['intent_pipeline'].predict([cleaned])[0]
    proba = models['intent_pipeline'].predict_proba([cleaned])[0]
    return intent, max(proba) * 100

def get_conv_response(text):
    cleaned = clean_text_light(text)
    user_vec = models['conv_tfidf'].transform([cleaned])
    sims = cosine_similarity(user_vec, models['conv_vectors'])[0]
    top_indices = np.argsort(sims)[-3:][::-1]
    top_valid = [(i, sims[i]) for i in top_indices if sims[i] > 0]
    if top_valid:
        idx = random.choice(top_valid)[0]
        return models['conv_responses'][idx], sims[idx]
    return models['conv_responses'][np.argmax(sims)], 0.0

def extract_keywords(text):
    cleaned = clean_text(text)
    vec = models['tfidf_sentiment'].transform([cleaned])
    features = models['tfidf_sentiment'].get_feature_names_out()
    scores = vec.toarray()[0]
    ws = [(features[i], scores[i]) for i in range(len(scores)) if scores[i] > 0]
    ws.sort(key=lambda x: x[1], reverse=True)
    return [w for w, s in ws[:5]]

def get_pos_tags(text):
    tokens = word_tokenize(text)
    tags = pos_tag(tokens)
    return {'nouns': [w for w, t in tags if t.startswith('NN')],
            'verbs': [w for w, t in tags if t.startswith('VB')],
            'adjectives': [w for w, t in tags if t.startswith('JJ')]}

def get_bot_response(user_msg):
    sentiment = analyze_sentiment(user_msg)
    intent, intent_conf = classify_intent(user_msg)
    conv_reply, conv_score = get_conv_response(user_msg)
    keywords = extract_keywords(user_msg)
    pos = get_pos_tags(user_msg)

    label = sentiment['label']
    conf = sentiment['confidence']

    mood = conf / 100 if label == 'Positive' else (0.5 if label == 'Neutral' else 1 - (conf / 100))
    st.session_state.mood_scores.append(mood)

    shift = "first message"
    if len(st.session_state.mood_scores) >= 2:
        diff = st.session_state.mood_scores[-1] - np.mean(st.session_state.mood_scores[:-1])
        if diff > 0.15: shift = "improving"
        elif diff < -0.15: shift = "declining"
        else: shift = "steady"

    avg_mood = np.mean(st.session_state.mood_scores)
    if avg_mood > 0.65: mood_text = f"very positive ({avg_mood:.2f})"
    elif avg_mood > 0.5: mood_text = f"slightly positive ({avg_mood:.2f})"
    elif avg_mood > 0.35: mood_text = f"neutral ({avg_mood:.2f})"
    else: mood_text = f"leaning negative ({avg_mood:.2f})"

    if intent in ('greeting', 'farewell', 'question') and conv_score > 0.3:
        response = conv_reply
    else:
        response = f"That's {label} ({conf:.1f}% confidence)."

    analysis = {'sentiment': sentiment, 'intent': intent, 'intent_conf': intent_conf,
                'conv_score': conv_score, 'keywords': keywords, 'pos': pos,
                'mood_summary': mood_text, 'mood_shift': shift}
    return response, analysis

if "messages" not in st.session_state:
    st.session_state.messages = []
if "mood_scores" not in st.session_state:
    st.session_state.mood_scores = []
if "analyses" not in st.session_state:
    st.session_state.analyses = []

st.markdown("""
<div class="header-gradient">
    <h1>NLP Sentiment Chatbot</h1>
    <p>Trained on 1.6 million tweets — powered by ML ensemble and NLP</p>
    <p class="author">Shria Biju &nbsp;|&nbsp; 23BCE10885</p>
    <div style="margin-top:12px;">
        <span class="tag">Voting Ensemble</span>
        <span class="tag">LogReg + SVM + NB</span>
        <span class="tag">TF-IDF Trigrams</span>
        <span class="tag">Negation Handling</span>
        <span class="tag">Cosine Similarity</span>
        <span class="tag">POS Tagging</span>
    </div>
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("### Shria Biju")
    st.markdown("**Reg No:** 23BCE10885")
    st.divider()
    st.markdown("### Models")
    st.markdown("""
    - **Sentiment**: Ensemble (LogReg + SVM + NB) + TF-IDF trigrams (100K tweets)
    - **Negation**: Contraction expansion + NOT_ prefix
    - **Intent**: Naive Bayes (6 classes)
    - **Conversation**: TF-IDF cosine similarity
    - **NLP**: POS tagging, TextBlob, keyword extraction
    """)
    st.divider()
    if st.session_state.mood_scores:
        st.markdown("### Mood Tracker")
        st.line_chart(st.session_state.mood_scores, color="#a78bfa")
        st.metric("Avg Mood", f"{np.mean(st.session_state.mood_scores):.2f}")
        st.caption(f"{len(st.session_state.mood_scores)} messages analyzed")
    st.divider()
    if st.button("Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.mood_scores = []
        st.session_state.analyses = []
        st.rerun()

for i, msg in enumerate(st.session_state.messages):
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
    if msg["role"] == "assistant" and i // 2 < len(st.session_state.analyses):
        a = st.session_state.analyses[i // 2]
        s = a['sentiment']
        if s['label'] == "Positive": tc = "pos-tag"
        elif s['label'] == "Negative": tc = "neg-tag"
        else: tc = "neu-tag"
        bar = "█" * int(s['confidence'] / 10) + "░" * (10 - int(s['confidence'] / 10))
        st.markdown(f"""
        <div class="analysis-card">
            <span class="label">Sentiment:</span> <span class="{tc}">{s['label']}</span> ({s['confidence']:.1f}%) <code>{bar}</code><br>
            <span class="label">Polarity:</span> {s['polarity']:+.3f} &nbsp;|&nbsp;
            <span class="label">Subjectivity:</span> {s['subjectivity']:.3f}<br>
            <span class="label">Intent:</span> {a['intent']} ({a['intent_conf']:.1f}%) &nbsp;|&nbsp;
            <span class="label">Match:</span> {a['conv_score']:.3f}<br>
            <span class="label">Keywords:</span> {', '.join(a['keywords'][:4]) if a['keywords'] else 'none'}<br>
            <span class="label">POS:</span> nouns={a['pos']['nouns'][:3]}, adj={a['pos']['adjectives'][:3]}<br>
            <span class="label">Mood:</span> {a['mood_summary']} ({a['mood_shift']})
        </div>
        """, unsafe_allow_html=True)

if prompt := st.chat_input("Type your message..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    response, analysis = get_bot_response(prompt)
    st.session_state.messages.append({"role": "assistant", "content": response})
    st.session_state.analyses.append(analysis)
    with st.chat_message("assistant"):
        st.markdown(response)
    s = analysis['sentiment']
    if s['label'] == "Positive": tc = "pos-tag"
    elif s['label'] == "Negative": tc = "neg-tag"
    else: tc = "neu-tag"
    bar = "█" * int(s['confidence'] / 10) + "░" * (10 - int(s['confidence'] / 10))
    st.markdown(f"""
    <div class="analysis-card">
        <span class="label">Sentiment:</span> <span class="{tc}">{s['label']}</span> ({s['confidence']:.1f}%) <code>{bar}</code><br>
        <span class="label">Polarity:</span> {s['polarity']:+.3f} &nbsp;|&nbsp;
        <span class="label">Subjectivity:</span> {s['subjectivity']:.3f}<br>
        <span class="label">Intent:</span> {analysis['intent']} ({analysis['intent_conf']:.1f}%) &nbsp;|&nbsp;
        <span class="label">Match:</span> {analysis['conv_score']:.3f}<br>
        <span class="label">Keywords:</span> {', '.join(analysis['keywords'][:4]) if analysis['keywords'] else 'none'}<br>
        <span class="label">POS:</span> nouns={analysis['pos']['nouns'][:3]}, adj={analysis['pos']['adjectives'][:3]}<br>
        <span class="label">Mood:</span> {analysis['mood_summary']} ({analysis['mood_shift']})
    </div>
    """, unsafe_allow_html=True)
    st.rerun()
