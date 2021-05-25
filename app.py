import streamlit as st
import numpy as np
import tensorflow as tf
import contractions
import re
import pickle
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

#function for data cleaning
def preprocessing(input_text):
    remove_url=re.sub(r'http\S+',' ',input_text) #removing the url
    remove_con=contractions.fix(remove_url)#for removing contractions like don't,I'd 
    remove_nump=re.sub('[^a-z-A-Z]',' ',remove_con) #removing numbers and punctuation
    remove_h=re.sub('[-*]',' ',remove_nump)#for removing any number of hyphens
    review_l=remove_h.lower()   #converting all characters into lowercase
    review_stop=[word for word in review_l.split() if (word not in stopwords.words('english')) and 
                 len(word)>1]#removing short words and stopwords
    lemmatized_words = [WordNetLemmatizer().lemmatize(word) for word in review_stop]#performing lemmatization
    review_text = " ".join(lemmatized_words)
    return review_text


@st.cache()#function for generating predictions
def prediction(text):
        model = tf.keras.models.load_model('saved_model.h5')    
        with open('tokenizer.pickle', 'rb') as handle:
               tokenizer = pickle.load(handle)
        cleaned_text=preprocessing(text)
        text_sequences=tokenizer.texts_to_sequences([cleaned_text])
        text_padded = pad_sequences(text_sequences, maxlen=252,padding='post')#252 as used during training of model
        pred=np.argmax(model.predict(text_padded))
        score=np.max(model.predict(text_padded))*100
        return pred,score
    

def main():
    # front end elements of the web page 
    html_temp = """ 
    <div style ="background-color:Red;padding:13px"> 
    <h1 style ="color:white;text-align:center;">Covid-19 Tweets Sentiment Analysis App</h1> 
    </div> 
    """
    st.set_page_config("Sentiment Analysis")
    
    st.markdown(html_temp,unsafe_allow_html=True)
    
    text=st.text_input('Enter Tweet')
    
    if st.button('Predict'):
        pred_value,score=prediction(text)
        if pred_value==2:
            st.success("Sentiment is Positive")
            st.success("Score: {}".format(round(score)))
        elif pred_value==1:
            st.info("Sentiment is Neutral")
            st.info("Score: {}".format(round(score)))
        else:
            st.error("Sentiment is Negative")
            st.error("Score: {}".format(round(score)))
            
if __name__ == '__main__':
      main()
