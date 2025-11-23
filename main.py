import tensorflow as tf
from keras.models import load_model
from keras.metrics import Precision, Recall
import numpy as np
from data.data_processor_v2 import load_and_prepare_data
from src.model_builder_v2 import create_model, train_model

# Config
data_path = "data/Cell_Phones_and_Accessories_5.json"
vocab_size = 10000
max_len = 120
embedding_dim = 128
epochs = 10
batch_size = 64

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("--- [PHASE 1] Loading and Preparing Data... ---")
    print("=" * 60)

    X_train, X_val, X_test, y_train, y_val, y_test = load_and_prepare_data(
        data_path, vocab_size, max_len
    )

    print(f"\nData shapes:")
    print(f"  Train: {X_train.shape}, Labels: {y_train.shape}")
    print(f"  Val:   {X_val.shape}, Labels: {y_val.shape}")
    print(f"  Test:  {X_test.shape}, Labels: {y_test.shape}")
    print("\n--- [PHASE 1] Data Preparation Complete. ---\n")

    print("=" * 60)
    print("--- [PHASE 2] Creating and Training Model... ---")
    print("=" * 60)

    model = create_model(vocab_size, max_len, embedding_dim)
    history = train_model(model, X_train, y_train, X_val, y_val, epochs, batch_size)

    print("\n--- [PHASE 2] Model Training Complete. ---\n")

    print("=" * 60)
    print("--- [PHASE 3] Evaluating Model on Test Data... ---")
    print("=" * 60)

    best_model = load_model(
        "sentiment_model_v3.keras",
    )

    results = best_model.evaluate(X_test, y_test)

    test_loss = results[0]
    test_acc = results[1]

    print("\n" + "=" * 60)
    print("--- [FINAL RESULTS] ---")
    print("=" * 60)
    print(f"Test Loss:      {test_loss:.4f}")
    print(f"Test Accuracy:  {test_acc*100:.2f}%")
    # print(f"Test Precision: {test_prec:.4f}")
    # print(f"Test Recall:    {test_rec:.4f}")
    print("=" * 60 + "\n")
