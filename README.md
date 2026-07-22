# AI / ML Project Portfolio

A single repository containing all AI/ML projects, each in its own
self-contained subfolder with its own README.

## Index

| # | Project |
|---|---|
| 1 | [Adult Census Income](./1-adult-census-income) |
| 2 | [Face Recognition](./2-face-recognition) |
| 3 | [CIFAR-10 Image Classification](./3-cifar10) |
| 4 | [Cancer Detection](./4-cancer-detection) |
| 5 | [NLP Chatbot](./5-nlp-chatbot) |
| 6 | [LunarLander (Reinforcement Learning)](./6-lunarlander) |
| 7 | [CartPole (Reinforcement Learning)](./7-cartpole) |
| 8 | [Iris Classification](./8-iris-classification) |
| 9 | [Movie Recommendation System](./9-movie-recommendation) |
| 10 | [RAG Capstone Project](./10-rag-capstone-project) |

Each project folder is self-contained with its own `README.md` covering
objective, dataset, methodology, results, and conclusion (or setup/run
instructions for the app-based projects).

## Structure

ai-ml-assignments/
├── README.md <- this file
├── 1-adult-census-income/
│ ├── adult.csv
│ ├── adult_census_assignment.ipynb
│ └── README.md
├── 2-face-recognition/
│ ├── face_recognition.ipynb
│ └── README.md
├── 3-cifar10/
│ ├── cifar10.ipynb
│ └── README.md
├── 4-cancer-detection/
│ ├── cancer_detection.ipynb
│ └── README.md
├── 5-nlp-chatbot/
│ └── ...
├── 6-lunarlander/
│ └── ...
├── 7-cartpole/
│ └── ...
├── 8-iris-classification/
│ └── ...
├── 9-movie-recommendation/
│ └── ...
├── 10-rag-capstone-project/
│ └── README.md
└── .gitignore


Each project folder is self-contained with its own `requirements.txt` (or
equivalent) and `README.md`, so every one is understandable and runnable
on its own even though they all live under one repo link.

## Setup

Each project folder has its own dependencies and run instructions — see
that folder's `README.md`. In general:

```bash
cd <project-folder>
pip install -r requirements.txt   # or the packages listed in its README
```

Notebooks (`.ipynb`) can be opened directly in VSCode with the Jupyter
extension; app-based projects (Flask/Streamlit) are run with `python
app.py` or `streamlit run app.py` as noted in their README.