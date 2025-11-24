import pickle
import numpy as np
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
from data.data_processor_v2 import clean_text

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

    print(f"\n[DEBUG] Original: {text_input}")
    print(f"[DEBUG] Cleaned:  '{cleaned_text}'")
    print(f"[DEBUG] Tokens:   {cleaned_text.split()}")

    sequence = tokenizer.texts_to_sequences([cleaned_text])
    print(f"[DEBUG] Sequence: {sequence}")

    padded_sequence = pad_sequences(sequence, maxlen=MAX_LEN)

    prediction_probs = model.predict(padded_sequence, verbose=0)
    score = prediction_probs[0][0]

    if score > 0.5:
        predicted_class_index = 1  # ПОМЕНЯЛ!
        confidence = score * 100
    else:
        predicted_class_index = 0  # ПОМЕНЯЛ!
        confidence = (1 - score) * 100

    predicted_class_name = class_names[predicted_class_index]

    print(f"   -> Vorhersage: {predicted_class_name} (Konfidenz: {confidence:.2f}%)")
    print(f"   -> (Debug: Raw Score: {score:.4f})")

    print(prediction_probs)
