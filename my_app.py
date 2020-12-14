import streamlit as st
import numpy as np
import pandas as pd
import pickle
from nltk.stem import SnowballStemmer
from nltk.tokenize import word_tokenize

st.subheader('Clasificador de Comentarios en Redes Sociales')

comentario = st.text_input('Comentario:')

# Load classification model
with open('./modelo.pkl', 'rb') as modelo:
        classifier = pickle.load(modelo)

with open('./vectorizer.pkl', 'rb') as vectorizador:
        vect = pickle.load(vectorizador)

# Función de preprocesamiento
def stemfraseesp(frase):    
    token_words=word_tokenize(frase)
    token_words
    stem_sentence=[]    
    spanishStemmer=SnowballStemmer("spanish",ignore_stopwords=True)
    for word in token_words:
        stem_sentence.append(spanishStemmer.stem(word))
        stem_sentence.append(" ")
    return "".join(stem_sentence)
        
if comentario != '':
    # Pre-process 
    #sentence = vect.transform(stemfraseesp(comentario)) 
    sentence=stemfraseesp(comentario)
    dato = [sentence]
    texto_vec=vect.transform(dato)
         
    # Make predictions
    with st.spinner('Predicting...'):
        clase=classifier.predict_proba(texto_vec)[:,0]>=0.3
        if clase == True:
            a="Negativo"
        else:
            a="Positivo"
        
     #Show predictions
    st.write('Prediction:',a)
        
