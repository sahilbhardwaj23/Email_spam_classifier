import streamlit as st
import pandas as pd
import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from gensim.models import Word2Vec

# Download necessary NLTK resources
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')

# Load the pre-trained model
model_file_path = 'random_forest__baseline_model.pkl'  # Change this to your actual file path
with open(model_file_path, 'rb') as model_file:
    model = pickle.load(model_file)

# Function for text preprocessing and embedding
def preprocess_and_embed(text: str) -> pd.DataFrame:
    stop_words = set(stopwords.words('english')) - {'not', 'but', 'however', 'no', 'yet'}
    
    df = pd.DataFrame({'user_text': [text]})
    df['text'] = df['user_text'].str.lower()
    df['text'] = df['text'].str.strip()
    df['text'] = df['text'].str.replace('\n', ' ', regex=True)
    df['word_count'] = df['text'].apply(lambda x: len(x.split()))
    df['num_stop_words'] = df['text'].apply(lambda x: len([word for word in x.split() if word in stop_words]))
    df['num_chars'] = df['text'].apply(len)
    df['num_punctuation_chars'] = df['text'].apply(lambda x: sum([1 for char in x if char in '.,!?;:"\'()[]{}-']))
    df['text'] = df['text'].apply(lambda x: re.sub(r'[^A-Za-z0-9\s!?.,]', '', str(x)))
    df['text'] = df['text'].apply(lambda x: ' '.join([word for word in x.split() if word.lower() not in stop_words]))

    lemmatizer = WordNetLemmatizer()
    df['text'] = df['text'].apply(lambda x: ' '.join([lemmatizer.lemmatize(word) for word in x.split()]))
    df['text'] = df['text'].apply(lambda x: re.sub(r'http\S+|www\S+|https\S+', '', x, flags=re.MULTILINE))
    df['text'] = df['text'].apply(lambda x: re.sub(r'\S+@\S+', '', x))
    df['text'] = df['text'].apply(lambda x: re.sub(r'<.*?>', '', x))
    df['text'] = df['text'].apply(lambda x: re.sub(r'[^a-zA-Z\s]', '', x))
    
    df['word_count_after_preprocessing'] = df['text'].apply(lambda x: len(x.split()))
    df['num_chars_after_preprocessing'] = df['text'].apply(len)
    df['tokenized_text'] = df['text'].apply(lambda text: word_tokenize(text) if isinstance(text, str) else [])

    sentences = df['tokenized_text'].to_list()
    model_w2v = Word2Vec(sentences, vector_size=100, window=5, min_count=1, workers=4)
    
    df['word2vec_embedded_vector'] = df['tokenized_text'].apply(
        lambda tokens: (
            [sum(x) / len(x) if len(x) > 0 else 0 for x in zip(*[model_w2v.wv[word] for word in tokens if word in model_w2v.wv])] 
            ) if len(tokens) > 0 else [0] * model_w2v.vector_size
        )
    
    df = df.drop(columns=['text'])
    expanded_df = pd.DataFrame(df['word2vec_embedded_vector'].tolist())
    expanded_df.columns = [f'vector_dim_{i+1}' for i in range(expanded_df.shape[1])]
    final_df = pd.concat([df, expanded_df], axis=1)
    final_df.drop(columns=['user_text', 'tokenized_text', 'word2vec_embedded_vector'], inplace=True)
    
    return final_df

# Streamlit app
def main():
    st.title("Email Spam Detection")

    # Input text box for user to input the email text
    user_input = st.text_area("Enter the email text to classify:", "")

    # Button to trigger classification
    if st.button("Classify Email"):
        if user_input:
            # Preprocess and embed the input text
            result_df = preprocess_and_embed(user_input)
            
            # Make a prediction
            prediction = model.predict(result_df)
            
            # Output result
            if prediction == 1:
                st.success("This email is classified as: **Spam**")
            else:
                st.success("This email is classified as: **Not Spam**")
        else:
            st.warning("Please enter some text for classification.")

if __name__ == "__main__":
    main()
