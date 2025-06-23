# 🛍️ Amazon Customer Reviews Sentiment Analysis

This project performs sentiment analysis on Amazon product reviews using Natural Language Processing (NLP). The goal is to classify customer sentiments as **Positive**, **Negative**, or **Neutral** to better understand user satisfaction and feedback patterns.

## 🎯 Objectives

- Clean and preprocess Amazon review text data
- Analyze word usage and review trends
- Train a machine learning model to classify sentiment
- Evaluate model performance using accuracy, precision, recall, and F1-score

## 🧰 Technologies & Libraries

- **Python**
- **Pandas** – Data handling
- **NumPy** – Numerical computations
- **Matplotlib & Seaborn** – Visualizations
- **NLTK / SpaCy** – Text preprocessing
- **Scikit-learn** – ML model training & evaluation
- *(Optional)* **Transformers** – for BERT or other deep learning-based sentiment models

## 📄 Dataset Overview

- **Source**: Amazon product reviews dataset  
- **Columns**:  
  - `Review` – Customer feedback  
  - `Sentiment` – Label (Positive, Negative, Neutral)  

## 🧹 Text Preprocessing

- Removed punctuation, numbers, and special characters
- Converted to lowercase
- Removed stopwords
- Tokenized and lemmatized words
- Vectorized using TF-IDF or CountVectorizer

## 🧠 Models Used

- Logistic Regression
- Naive Bayes
- Support Vector Machine (SVM)
- *(Optional)* BERT for advanced sentiment classification

## 📊 Evaluation Metrics

- Accuracy
- Confusion Matrix
- Precision, Recall, F1 Score
- ROC Curve (for binary sentiment tasks)

## 🔍 Sample Insights

- Majority of reviews are positive, followed by negative
- Frequent keywords in negative reviews: *“bad”*, *“worst”*, *“return”*
- Positive reviews highlight *“great quality”*, *“fast delivery”*

## 📈 Visualizations

- Word clouds for each sentiment
- Sentiment distribution pie chart
- Review length vs sentiment
- Model performance comparison bar chart

## 🚀 How to Run

```bash
# Clone the repo
git clone https://github.com/vasu0021/amazon-sentiment-analysis.git
cd amazon-sentiment-analysis

# (Optional) Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the Jupyter Notebook
jupyter notebook Amazon_Sentiment_Analysis.ipynb
