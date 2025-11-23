import pickle
import numpy as np
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
from data.data_processor import clean_text

MAX_LEN = 120

# MODEL_PATH = 'sentiment_model.keras'     # V1 (72.01%)
MODEL_PATH = "sentiment_model_v3.keras"  # V2 (71.89%)
TOKENIZER_PATH = "tokenizer.pickle"

print(f"Lade Modell von {MODEL_PATH}...")
model = load_model(MODEL_PATH)

print(f"Lade Tokenizer von {TOKENIZER_PATH}...")

with open(TOKENIZER_PATH, "rb") as handle:
    tokenizer = pickle.load(handle)

class_names = ["Negativ", "Positiv"]

print("\n--- Modell bereit. Geben Sie 'exit' oder 'quit' ein, um zu beenden ---")

while True:
    text_input = input("\nGeben Sie Ihre Bewertung ein: ")

    if text_input.lower() in ["exit", "quit"]:
        print("Beende...")
        break

    cleaned_text = clean_text(text_input)

    sequence = tokenizer.texts_to_sequences([cleaned_text])

    padded_sequence = pad_sequences(sequence, maxlen=MAX_LEN)

    prediction_probs = model.predict(padded_sequence, verbose=0)

    

    predicted_class_index = np.argmax(prediction_probs[0])

    predicted_class_name = class_names[predicted_class_index]

    confidence = np.max(prediction_probs[0]) * 100

    print(f"   -> Vorhersage: {predicted_class_name} (Konfidenz: {confidence:.2f}%)")
    print(f"   -> (Debug: Wahrscheinlichkeiten [Neg, Pos]: {prediction_probs[0]})")
    print(prediction_probs)
