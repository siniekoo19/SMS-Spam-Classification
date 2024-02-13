import streamlit as st
from streamlit_extras.let_it_rain import rain
import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import os 
os.system('sudo pip install scikit-learn') 

cv = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

def transfrom_text(t):
    text = re.sub('[^a-zA-Z0-9]', ' ', t)
    text = text.replace('"', '')
    text = text.lower()
    text = nltk.word_tokenize(text)
    ps = PorterStemmer()
    text = [ps.stem(word) for word in text if not word in stopwords.words('english')]
    text = ' '.join(text)
    
    return text 

st.title(":red[SMS Spam Classifier]:sunglasses:")

msg = st.text_area("Enter the Message : ")

if st.button('Predict'):
    # Preprocess
    transfrom_sms = transfrom_text(msg)

    # Vectorize
    vect_sms = cv.transform([transfrom_sms]) 

    # Predict
    Result = model.predict(vect_sms)[0]

    # Display
    if Result == 1:
        st.markdown("### The SMS is Spam :fearful:")
        rain( 
            emoji="😨", 
            font_size=40,  # the size of emoji 
            falling_speed=4,  # speed of raining 
            animation_length="infinite",  # for how much time the animation will happen 
        ) 
    elif Result == 0:
        st.markdown("### The SMS is not Spam :smile:")
        rain( 
            emoji="😄", 
            font_size=40,  # the size of emoji 
            falling_speed=4,  # speed of raining 
            animation_length="infinite",  # for how much time the animation will happen 
        ) 