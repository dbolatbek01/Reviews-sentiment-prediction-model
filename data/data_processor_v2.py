import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
import numpy as np

if not hasattr(np, "unicode_"):
    np.unicode_ = np.str_
from keras_preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
import pickle


nltk.download("stopwords")
stop_words = set(stopwords.words("english"))


def clean_text(text):
    text = str(text).lower()

    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"[^a-z\s]", "", text)

    tokens = text.split()

    new_tokens = []

    for word in tokens:
        if word not in stop_words and len(word) > 2:
            new_tokens.append(word)

    tokens = new_tokens

    return " ".join(tokens)


def load_and_prepare_data(json_path, vocab_size=10000, max_len=120):
    data = pd.read_json("data/Cell_Phones_and_Accessories_5.json", lines=True)
    data = data[["reviewText", "overall"]]

    class_mapping = {
        1.0: 0,
        2.0: 0,
        4.0: 1,
        5.0: 1,
    }

    data["sentiment"] = data["overall"].map(class_mapping)

    data_cleaned = data.dropna(subset=["sentiment"]).copy()
    data_cleaned["sentiment"] = data_cleaned["sentiment"].astype(int)

    min_class_size = data_cleaned["sentiment"].value_counts().min()

    data_balanced = pd.concat(
        [
            data_cleaned[data_cleaned["sentiment"] == 0].sample(
                min_class_size, random_state=2001
            ),
            data_cleaned[data_cleaned["sentiment"] == 1].sample(
                min_class_size, random_state=2001
            ),
        ]
    )

    print(data_balanced["sentiment"].value_counts())

    data_balanced["cleaned_text"] = data_balanced["reviewText"].apply(clean_text)

    tokenizer = Tokenizer(num_words=vocab_size, oov_token="<unk>")

    tokenizer.fit_on_texts(data_balanced["cleaned_text"])

    with open("tokenizer.pickle", "wb") as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

    sequences = tokenizer.texts_to_sequences(data_balanced["cleaned_text"])
    padded_sequences = pad_sequences(
        sequences, maxlen=max_len, padding="post", truncating="post"
    )

    X = padded_sequences
    y = data_balanced["sentiment"].values

    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=0.2, stratify=y_train_val
    )

    print("Data shapes:")
    print(f" Train: {X_train.shape}, Labels: {y_train.shape}")
    print(f" Val:  {X_val.shape}, Labels: {y_val.shape}")
    print(f" Test: {X_test.shape}, Labels: {y_test.shape}")

    return X_train, X_val, X_test, y_train, y_val, y_test


if __name__ == "__main__":
    # The path '../data/' means "go up one folder, then into data"
    # Adjust this path if your file is located elsewhere
    DATA_PATH = "../data/Cell_Phones_and_Accessories_5.json"

    print("--- [Test Run data_processor.py] ---")
    # Call the function to test it
    load_and_prepare_data(DATA_PATH, vocab_size=10000, max_len=120)
    print("--- [Test Run Complete] ---")
