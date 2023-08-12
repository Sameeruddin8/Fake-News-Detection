from flask import Flask, render_template, request
import pickle
import pandas 
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

app = Flask(__name__)

ps = PorterStemmer()

def transform(text):
    text = text.lower()
    text = nltk.word_tokenize(text)
    y = []
    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)
    text = y[:]
    y.clear()
    for i in text:
        y.append(ps.stem(i))
    return " ".join(y)

model = pickle.load(open('model.pkl', 'rb'))
vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))

@app.route("/")
def home():
    return render_template('index.html')

@app.route("/predict", methods = ['GET', 'POST'])
def predict():
    text = transform(request.form.get('text'))
    vector = vectorizer.transform([text])
    result = model.predict(vector)
    # probabilities = model.predict_proba(vector)[0]
    # confidence = max(probabilities) * 100
    if result == 0:
        prediction = "Fake"
    else:
        prediction = "Not Fake"

    return render_template('index.html', predicteeeed=prediction)
if __name__ == '__main__':
    app.run(debug = True)