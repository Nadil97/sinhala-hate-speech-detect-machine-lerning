import nltk
nltk.download('punkt')

from flask import Flask, render_template, request
import joblib
import re
from nltk.tokenize import word_tokenize

app = Flask(__name__)

# Load trained components
model = joblib.load('sinhala_hate_model.pkl')
vectorizer = joblib.load('sinhala_vectorizer.pkl')
label_encoder = joblib.load('sinhala_label_encoder.pkl')

sinhala_stopwords = set([
    'ඔහු', 'ඇය', 'ඔබ', 'අපි', 'මම', 'එය', 'මෙය', 'ඒක', 'දෙවියන්', 'නැහැ', 'ඔව්'
])

def preprocess(text):
    text = str(text).lower()
    text = re.sub(r'[^\u0D80-\u0DFF\s]', '', text)
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in sinhala_stopwords]
    return " ".join(tokens)

@app.route('/', methods=['GET', 'POST'])
def index():
    result_message = None
    user_text = ""

    if request.method == 'POST':
        user_text = request.form['comment']
        cleaned_text = preprocess(user_text)
        vector = vectorizer.transform([cleaned_text])
        pred = model.predict(vector)[0]  # single prediction (int)
        label_str = str(label_encoder.inverse_transform([pred])[0])  # safe string

        if label_str == '1':  # ← if '1' means hate
             result_message = "***** is a bad word"
        else:
            result_message = f"\"{user_text}\" is not a hate word"



    return render_template('index.html', result=result_message)

if __name__ == '__main__':
    app.run(debug=True)
