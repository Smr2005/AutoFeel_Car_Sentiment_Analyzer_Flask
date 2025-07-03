# 🚗 AutoFeel: Car Sentiment Analyzer

*AutoFeel* is a Flask-based web application that allows users to submit and explore car reviews. It performs sentiment analysis on user reviews using a machine learning model and provides insights for different brands and models.

✅ Built with:
- Python (Flask, Pandas, TextBlob/NLTK)
- Machine Learning (Scikit-learn)
- HTML/CSS (Jinja2 templates)
- 🚫 No SQL/Database required (uses CSV as backend)

---

## 🔍 Features

- 📝 Submit car reviews with brand, model, rating, and opinion
- 💬 Auto-analyzes sentiment (Positive, Negative, Neutral)
- 📊 Explore sentiment trends and average ratings by brand/model
- 🔧 Pretrained ML model for more accurate predictions (optional)
- 💾 Uses a simple CSV file (car_reviews.csv) for data storage

---

## 📂 Project Structure

AutoFeel_Car_Sentiment_Analyzer_Flask/ ├── app.py                         # Main Flask app ├── requirements.txt              # Python dependencies ├── dataset/ │   └── car_reviews.csv           # All stored reviews ├── templates/                    # HTML templates │   ├── index.html │   ├── about.html │   ├── review_submit.html │   ├── explore_reviews.html │   └── manufacturer_insights.html ├── static/                       # CSS / JS / images (optional) │   └── styles.css └── sentiment_model.pkl          # ML model (optional)

---

## 🚀 Deployment Guide (Render.com)

### ✅ Step 1: Fork this repo to your GitHub

### ✅ Step 2: Deploy to Render
1. Create a free account on [https://render.com](https://render.com)
2. Click *"New Web Service"*
3. Connect your GitHub repo
4. Set the following values:
   - *Build Command*: (leave blank)
   - *Start Command*: gunicorn app:app
5. Set environment to *Python 3.x*
6. Click *Deploy*

### Optional: Use a render.yaml file

```yaml
services:
  - type: web
    name: autofeel-app
    env: python
    buildCommand: ""
    startCommand: gunicorn app:app
    plan: free


---

📦 Installation (Local Development)

# Clone the repo
git clone https://github.com/your-username/AutoFeel_Car_Sentiment_Analyzer_Flask.git
cd AutoFeel_Car_Sentiment_Analyzer_Flask

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

# Install dependencies
pip install -r requirements.txt

# Run the app
python app.py


---

🧠 How Sentiment Analysis Works

User review is cleaned using NLTK (stopwords, lemmatization)

Passed into a trained classifier (like Logistic Regression or Naive Bayes)

Prediction and confidence score are returned

Review + sentiment are appended to the car_reviews.csv



---

⚠️ Limitations

Render free tier has ephemeral filesystem: newly submitted reviews won't persist after app restarts.

For persistent storage: use PostgreSQL, SQLite, or external file services (like AWS S3).



---

🧪 Optional Improvements

🔁 Switch from CSV to PostgreSQL for long-term storage

📈 Add charts using Chart.js or Plotly for visual sentiment

🛡️ Add admin panel to moderate reviews

💬 Add user authentication for secure submissions

🌍 Add multi-language review support



---

🧾 License

This project is licensed under the MIT License.


---

✨ Acknowledgements

Built by [Your Name] | Powered by Flask + ML | Deployed on Render.com

---

### ✅ Bonus:
Let me know if you'd like:
- A *dark-mode preview badge* (for your repo README)
- Or a *demo video thumbnail* + embed template
