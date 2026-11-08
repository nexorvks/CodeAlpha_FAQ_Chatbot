# CodeAlpha FAQ Chatbot

An NLP-powered FAQ chatbot developed for the CodeAlpha Data Science Internship. This project uses Natural Language Processing to understand user intent and match it with the most relevant technical answers.

## Features
* **Linguistic Preprocessing:** Utilizes `NLTK` for tokenization, stop-word removal, and lemmatization.
* **Intent Matching:** Employs `Scikit-Learn`'s TF-IDF Vectorizer and Cosine Similarity to calculate the exact mathematical overlap between user input and the dataset.
* **Dynamic Fallbacks:** Gracefully handles unknown queries by offering the top 3 closest matching suggestions based on confidence thresholds.

## Usage
1. Install dependencies: `pip install -r requirements.txt`
2. Run the chatbot: `python nlp_chatbot.py`