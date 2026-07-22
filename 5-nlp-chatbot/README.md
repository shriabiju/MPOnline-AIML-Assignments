# NLP Sentiment Chatbot

A chatbot that uses NLP and Machine Learning to analyze sentiment, classify intent, and hold basic conversations.

**Models used**
- Sentiment: Voting Ensemble (Logistic Regression + SVM + Naive Bayes) trained on 100K tweets from the Sentiment140 dataset (1.6M tweets)
- Intent classification: Multinomial Naive Bayes (6 classes)
- Conversational response: TF-IDF cosine similarity retrieval

**Stack:** Python, scikit-learn, NLTK, TextBlob, Streamlit

**Live demo:** https://nlp-sentiment-chatbot.onrender.com
*(free-tier instance — spins down after inactivity, first load can take 30-60s)*

**Repo:** https://github.com/shriabiju/nlp-chatbot

---

## Project structure

```
nlp-chatbot/
├── app.py              # Streamlit app (loads trained models, runs the chatbot UI)
├── train_model.py       # Downloads data, cleans it, trains models, saves them
├── requirements.txt
├── render.yaml           # Render deployment config
├── .gitignore
├── chatbot_models/       # created after you run train_model.py (committed to git)
└── README.md
```

---

## 1. Set up locally in VS Code

**Prerequisites:** Python 3.10+ installed, VS Code with the Python extension.

```bash
# open this folder in VS Code, then in the integrated terminal:

# create and activate a virtual environment
python -m venv venv
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# install dependencies
pip install -r requirements.txt
```

In VS Code, select the `venv` interpreter: `Ctrl+Shift+P` → "Python: Select Interpreter" → choose the one inside `venv/`.

## 2. Train the models

This downloads the Sentiment140 dataset (~80MB) and trains all three models. It takes a few minutes.

```bash
python train_model.py
```

When it finishes you'll have a `chatbot_models/` folder containing the `.pkl` files the app needs. You only need to do this once (re-run it if you change the training code).

## 3. Run the app locally

```bash
streamlit run app.py
```

This opens the chatbot at `http://localhost:8501`.

---

## 4. Push to GitHub

Code lives at: **https://github.com/shriabiju/nlp-chatbot** (branch: `main`)

For future changes, the usual flow is:

```bash
git add .
git commit -m "your message"
git push
```

**Important:** `.gitignore` excludes the raw dataset (`sentiment140.csv`) since it's large and regenerable, but it does **not** exclude `chatbot_models/`. Those `.pkl` files need to stay committed — Render loads them directly at deploy time, and retraining on every deploy would be slow and unreliable on a free tier.

If you ever re-clone this repo fresh, set the remote with:

```bash
git remote add origin https://github.com/shriabiju/nlp-chatbot.git
```

---

## 5. Deploy on Render

**Status: deployed.** The service `nlp-sentiment-chatbot` is live at https://nlp-sentiment-chatbot.onrender.com, set up via Render's **Blueprint** flow (using the included `render.yaml`), connected to `shriabiju/nlp-chatbot` on the `main` branch.

Render auto-deploys on every push to `main` — no need to manually redeploy after `git push`, though you can trigger one manually from the service dashboard (**Manual Deploy** button) if needed.

**Note on the free plan:** the free instance spins down after ~15 minutes of inactivity and takes 30-60 seconds to wake back up on the next request — that's expected, not a bug. The dashboard's "Upgrade now" banner is about avoiding that cold start, not required for the app to work.

If you're setting this up from scratch on a new Render account:
1. [render.com](https://render.com) → sign in with GitHub
2. **New +** → **Blueprint** → select the repo → Render reads `render.yaml` → **Apply**