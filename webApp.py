import streamlit as st
import pickle
from PIL import Image
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()

def transform_text(text):
  text = text.lower()
  text = nltk.word_tokenize(text)
#removing the special characters
  y=[]
  for i in text:
    if i.isalnum():
      y.append(i)
#removing the stop words
  text = y[:]
  y.clear()
  for i in text:
    if i not in stopwords.words('english')and i not in string.punctuation:
      y.append(i)
#performing the stemming
  text = y[:]
  y.clear()
  for i in text:
    y.append(ps.stem(i))

  return " ".join(y)

st.title('Check Email/SMS spam or not')
image = Image.open('image.jpg')

st.image(image, caption='Are you irritate with spam SMS/EMAIL')
tfidf = pickle.load(open('vectorizer.pkl','rb'))
model = pickle.load(open('model.pkl','rb'))


input_sms = st.text_area('Enter the message')

if st.button('Predict'):
    #1 preprocess
    transform_sms = transform_text(input_sms)
    #2 vectorize
    vector_input = tfidf.transform([transform_sms])
    #3 predict
    result = model.predict(vector_input)[0]
    #4 Display
    if result == 1:
        st.header('Spam')
    else:
        st.header('Not spam')
        

