import tesnorflow as tf
import streamlit as st




model = tf.keras.models.load_model('simple_rnn_imdb.h5')

load_imdb_index = tf.keras.datasets.imdb.get_word_index()
reverse_imdb_index = dict([(value, key) for (key, value) in load_imdb_index.items()])


def decode_review(text):
    return ' '.join([reverse_imdb_index.get(i - 3, '?') for i in text])

def predict_review(text):
    encoded_text = [1] + [load_imdb_index.get(word, 2) for word in text.split()]
    encoded_text = tf.keras.preprocessing.sequence.pad_sequences([encoded_text], value=0, maxlen=500)
    prediction = model.predict(encoded_text)
    return prediction[0][0]


st.title("IMDB Movie Review Sentiment Analysis")
st.write("Enter a movie review to predict its sentiment (positive or negative).")

review_input = st.text_area("Movie Review", height=200)
if st.button("Predict Sentiment"):
    if review_input:
        sentiment_score = predict_review(review_input)
        if sentiment_score > 0.5:
            st.success(f"Positive Review (Score: {sentiment_score:.2f})")
        else:
            st.error(f"Negative Review (Score: {sentiment_score:.2f})")
    else:
        st.warning("Please enter a movie review to predict its sentiment.")