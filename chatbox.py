# ============================================================
# CODEALPHA DATA SCIENCE INTERNSHIP
# TASK: NLP FAQ CHATBOT
# ============================================================

import logging
from pathlib import Path
import pandas as pd
import numpy as np

# NLP and Machine Learning Libraries
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ============================================================
# SETUP & NLTK DOWNLOADS
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s: %(message)s"
)

# Download necessary NLTK data silently
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('tokenizers/punkt_tab')
    nltk.data.find('corpora/stopwords')
    nltk.data.find('corpora/wordnet')
except LookupError:
    logging.info("Downloading required NLTK linguistic datasets...")
    nltk.download('punkt', quiet=True)
    nltk.download('punkt_tab', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)

BASE_DIR = Path(__file__).resolve().parent
FAQ_FILE = BASE_DIR / "faqs.csv"

# ============================================================
# NLP CHATBOT ENGINE
# ============================================================

class NLPChatbot:
    def __init__(self, csv_path=FAQ_FILE, confidence_threshold=0.25):
        self.csv_path = Path(csv_path)
        self.threshold = confidence_threshold
        
        # NLP Tools
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        self.vectorizer = TfidfVectorizer()
        
        # Data and Vectors
        self.df = self.load_data()
        self.faq_vectors = self.train_tfidf()

    def load_data(self):
        """Loads and validates the faqs.csv dataset."""
        if not self.csv_path.exists():
            raise FileNotFoundError(f"Cannot find {self.csv_path.name}. Please ensure it is in the same directory.")
            
        try:
            df = pd.read_csv(self.csv_path)
        except Exception as e:
            raise ValueError(f"Failed to read CSV: {e}")

        df.columns = df.columns.str.strip().str.lower()
        if not {'question', 'answer'}.issubset(df.columns):
            raise ValueError("The dataset must contain 'question' and 'answer' columns.")
            
        df = df.dropna(subset=['question', 'answer'])
        logging.info(f"Successfully loaded {len(df)} FAQs.")
        return df

    def preprocess_text(self, text):
        """Applies NLTK tokenization, stop-word removal, and lemmatization."""
        # 1. Tokenize (split into words) and convert to lowercase
        tokens = word_tokenize(str(text).lower())
        
        # 2. Remove punctuation and stop words, then apply Lemmatization (reduce to root word)
        clean_tokens = [
            self.lemmatizer.lemmatize(word) 
            for word in tokens 
            if word.isalnum() and word not in self.stop_words
        ]
        
        return " ".join(clean_tokens)

    def train_tfidf(self):
        """Converts the text questions into TF-IDF mathematical vectors."""
        # Preprocess all questions in the dataset
        self.df['processed_question'] = self.df['question'].apply(self.preprocess_text)
        
        # Fit the vectorizer and transform questions into a numerical matrix
        vectors = self.vectorizer.fit_transform(self.df['processed_question'])
        return vectors

    def get_response(self, user_input):
        """Uses Cosine Similarity to match user input to the best FAQ."""
        if not user_input.strip():
            return "Please type a question."

        # Preprocess and vectorize the user's input
        processed_input = self.preprocess_text(user_input)
        input_vector = self.vectorizer.transform([processed_input])

        # Calculate cosine similarity between the input and all FAQ questions
        similarities = cosine_similarity(input_vector, self.faq_vectors).flatten()
        
        # Find the index of the highest similarity score
        best_match_idx = np.argmax(similarities)
        best_score = similarities[best_match_idx]

        if best_score < self.threshold:
            # Fallback intent matching for low confidence
            top_3_idx = similarities.argsort()[-3:][::-1]
            suggestions = "\n".join([f"- {self.df['question'].iloc[i]}" for i in top_3_idx if similarities[i] > 0.05])
            
            response = "I couldn't find an exact match. Did you mean one of these?\n" + suggestions
            if not suggestions:
                response = "I'm sorry, I don't have information on that topic."
            return response
            
        # Return the best matching answer and its mathematical confidence score
        answer = self.df['answer'].iloc[best_match_idx]
        return f"{answer}\n[Intent Confidence: {best_score:.0%}]"


# ============================================================
# USER INTERFACE
# ============================================================

def main():
    print("=" * 60)
    print(" " * 15 + "NLP-POWERED FAQ CHATBOT")
    print("=" * 60)
    print("Initializing NLP engine. This may take a moment...\n")
    
    try:
        bot = NLPChatbot()
    except Exception as e:
        logging.error(f"Setup Error: {e}")
        return

    print("\nBot is ready! Type your question below.")
    print("(Type 'help' for examples, or 'quit' to exit)\n")

    while True:
        try:
            user_question = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nBot: Session terminated. Goodbye!")
            break
            
        if user_question.lower() in {'quit', 'exit', 'bye'}:
            print("Bot: Goodbye!")
            break
            
        if user_question.lower() == 'help':
            print("Bot: Here are a few things you can ask me:")
            sample_faqs = bot.df['question'].head(4).tolist()
            for q in sample_faqs:
                print(f"  - {q}")
            continue

        response = bot.get_response(user_question)
        print(f"Bot: {response}\n")


if __name__ == "__main__":
    main()