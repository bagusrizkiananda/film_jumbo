import streamlit as st
import pandas as pd
import re
import string
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, accuracy_score

# --- Fungsi preprocessing ---
def clean_text(text):
    text = text.lower()
    text = re.sub(r'\n', ' ', text)
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

# --- Load data ---
@st.cache_data
def load_data():
    df = pd.read_csv("film_jumbo.csv")
    df.dropna(subset=['ulasan', 'label'], inplace=True)
    df['ulasan_bersih'] = df['ulasan'].apply(clean_text)
    return df

# --- Model training ---
@st.cache_resource
def train_model(data):
    X = data['ulasan_bersih']
    y = data['label']
    
    vectorizer = TfidfVectorizer()
    X_vect = vectorizer.fit_transform(X)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_vect, y, test_size=0.2, random_state=42)
    
    model = MultinomialNB()
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    
    return model, vectorizer, acc

# --- Aplikasi Streamlit ---
def main():
    st.title("Analisis Sentimen Film Pabrik Gula (2025)")
    st.write("Model: Naive Bayes berbasis Text Mining (Tanpa File PKL)")
    
    data = load_data()
    model, vectorizer, acc = train_model(data)
    
    st.subheader("Akurasi Model")
    st.success(f"{acc*100:.2f}%")
    
    st.subheader("Uji Sentimen Baru")
    input_text = st.text_area("Masukkan ulasan film:")
    
    if st.button("Prediksi"):
        cleaned = clean_text(input_text)
        vect_input = vectorizer.transform([cleaned])
        prediction = model.predict(vect_input)[0]
        
        st.write("**Hasil Prediksi Sentimen:**")
        if prediction == 'positif':
            st.success("Positif")
        elif prediction == 'negatif':
            st.error("Negatif")
        else:
            st.info(prediction)

if __name__ == "__main__":
    main()
