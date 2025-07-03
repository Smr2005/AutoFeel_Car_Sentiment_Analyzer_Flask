from flask import Flask, render_template, request, jsonify
import pandas as pd
import pickle
import os
import re
import string
import nltk

# NLTK downloads (only once)
nltk.download('stopwords')
nltk.download('wordnet')
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# App setup
app = Flask(__name__)
DATASET_PATH = "dataset/car_reviews.csv"

# Check dataset
if not os.path.exists(DATASET_PATH):
    raise FileNotFoundError("Dataset not found! Please ensure 'dataset/car_reviews.csv' exists.")

# Load trained sentiment model
with open("sentiment_model.pkl", "rb") as f:
    sentiment_model = pickle.load(f)

# --- Text Cleaning Function (same as train_model.py) ---
stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = [lemmatizer.lemmatize(w) for w in text.split() if w not in stop_words]
    return " ".join(tokens)

# --- Helper to get all reviews ---
def get_all_reviews():
    df = pd.read_csv(DATASET_PATH)
    df = df.dropna(subset=['Brand', 'Car', 'Review', 'Rating', 'Sentiment'])
    reviews = []
    for _, row in df.iterrows():
        reviews.append({
            'brand': row['Brand'],
            'car': row['Car'],
            'review': row['Review'],
            'rating': row['Rating'],
            'sentiment': row['Sentiment']
        })
    return reviews

# Home
@app.route('/')
def index():
    return render_template('index.html')

# About
@app.route('/about')
def about():
    return render_template('about.html')

# Submit Review
@app.route('/submit', methods=['GET', 'POST'])
def submit():
    if request.method == 'POST':
        brand = request.form['brand']
        car = request.form['car']
        review = request.form['review']
        rating = float(request.form['rating'])

        # Clean the text
        cleaned_review = clean_text(review)

        # Predict sentiment
        sentiment = sentiment_model.predict([cleaned_review])[0].capitalize()

        # Optional: Get confidence
        probas = sentiment_model.predict_proba([cleaned_review])[0]
        confidence = max(probas)  # Highest class probability
        sentiment += f" ({confidence:.2f})"  # e.g. "Positive (0.94)"

        # Append new review to CSV
        new_data = pd.DataFrame([{
            "Brand": brand,
            "Car": car,
            "Review": review,
            "Rating": rating,
            "Sentiment": sentiment
        }])
        new_data.to_csv(DATASET_PATH, mode='a', header=False, index=False)

        return render_template('review_submit.html', success=True)

    return render_template('review_submit.html', success=False)

# Get cars by brand
@app.route('/get_cars', methods=['POST'])
def get_cars():
    selected_brand = request.form.get('brand')
    all_reviews = get_all_reviews()

    filtered_cars = sorted(set(
        r['car'] for r in all_reviews if r['brand'] == selected_brand
    ))

    return jsonify({'cars': filtered_cars})

# Explore Reviews
@app.route('/explore_reviews', methods=['GET', 'POST'])
def explore_reviews():
    all_reviews = get_all_reviews()
    brands = sorted(set(r['brand'] for r in all_reviews))
    selected_brand = request.form.get('brand') or ''
    selected_car = request.form.get('car') or ''

    filtered_reviews = [
        r for r in all_reviews
        if (not selected_brand or r['brand'] == selected_brand) and
           (not selected_car or r['car'] == selected_car)
    ]

    avg_rating = sum(r['rating'] for r in filtered_reviews) / len(filtered_reviews) if filtered_reviews else 0

    # Count sentiment labels
    sentiment_counts = {
        'Positive': sum(1 for r in filtered_reviews if 'Positive' in r['sentiment']),
        'Negative': sum(1 for r in filtered_reviews if 'Negative' in r['sentiment']),
        'Neutral': sum(1 for r in filtered_reviews if 'Neutral' in r['sentiment'])
    }

    cars = sorted(set(r['car'] for r in all_reviews if r['brand'] == selected_brand)) if selected_brand else []

    return render_template(
        'explore_reviews.html',
        brands=brands,
        cars=cars,
        reviews=filtered_reviews,
        selected_brand=selected_brand,
        selected_car=selected_car,
        avg_rating=avg_rating,
        sentiment_counts=sentiment_counts
    )

# Manufacturer Insights
@app.route('/manufacturer_insights', methods=['GET', 'POST'])
def manufacturer_insights():
    df = pd.read_csv(DATASET_PATH)
    brands = sorted(df['Brand'].dropna().unique())
    cars = sorted(df['Car'].dropna().unique())
    cars_by_brand = df.groupby('Brand')['Car'].unique().apply(list).to_dict()

    filtered_reviews = []
    summary = None
    selected_brand = ""
    selected_car = ""

    if request.method == 'POST':
        selected_brand = request.form.get('brand')
        selected_car = request.form.get('car')

        filtered = df[
            (df['Brand'].str.lower() == selected_brand.strip().lower()) &
            (df['Car'].str.lower() == selected_car.strip().lower())
        ]

        filtered_reviews = filtered.to_dict(orient='records')

        if not filtered.empty:
            avg_rating = round(filtered['Rating'].mean(), 2)
            positive_count = filtered[filtered['Sentiment'].str.contains('Positive', case=False)].shape[0]
            negative_count = filtered[filtered['Sentiment'].str.contains('Negative', case=False)].shape[0]
            neutral_count = filtered[filtered['Sentiment'].str.contains('Neutral', case=False)].shape[0]
            total_reviews = filtered.shape[0]

            summary = {
                'avg_rating': avg_rating,
                'positive_count': positive_count,
                'negative_count': negative_count,
                'neutral_count': neutral_count,
                'total_reviews': total_reviews
            }

    return render_template(
        'manufacturer_insights.html',
        brands=brands,
        cars=cars,
        cars_by_brand=cars_by_brand,
        filtered_reviews=filtered_reviews,
        summary=summary,
        selected_brand=selected_brand,
        selected_car=selected_car
    )

# Run App
if _name_ == '_main_':
    port = int(os.environ.get("PORT", 5000))  # Render sets PORT environment variable
    app.run(debug=True, host='0.0.0.0', port=port)
