import pandas as pd, re, string, pickle, nltk
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer

# Setup
nltk.download('stopwords')
nltk.download('wordnet')
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

# Load dataset
df = pd.read_csv("dataset/car_reviews.csv")
df = df[['Review', 'Rating']].dropna()

# Label sentiments
def label_sentiment(rating):
    if rating <= 2:
        return "negative"
    elif rating == 3:
        return "neutral"
    else:
        return "positive"

df["sentiment"] = df["Rating"].apply(label_sentiment)

# Clean text
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = [lemmatizer.lemmatize(w) for w in text.split() if w not in stop_words]
    return " ".join(tokens)

df["clean"] = df["Review"].apply(clean_text)

# Dynamically balance by minimum class count
min_class_count = df["sentiment"].value_counts().min()
df_balanced = pd.concat([
    df[df["sentiment"] == "positive"].sample(min_class_count, random_state=42),
    df[df["sentiment"] == "neutral"].sample(min_class_count, random_state=42),
    df[df["sentiment"] == "negative"].sample(min_class_count, random_state=42)
], axis=0).sample(frac=1, random_state=42)

X = df_balanced["clean"]
y = df_balanced["sentiment"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Model pipeline
model = Pipeline([
    ("tfidf", TfidfVectorizer(ngram_range=(1, 3), max_features=20000)),
    ("clf", LogisticRegression(max_iter=1000))
])

# Train model
model.fit(X_train, y_train)
print("✅ Train Accuracy:", model.score(X_train, y_train))
print("✅ Test Accuracy:", model.score(X_test, y_test))

# Save model
with open("sentiment_model.pkl", "wb") as f:
    pickle.dump(model, f)
print("✅ Model saved successfully!")
