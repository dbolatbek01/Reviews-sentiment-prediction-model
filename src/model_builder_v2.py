import tensorflow as tf

from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional
from keras.callbacks import EarlyStopping


def create_model(vocab_size, max_len, embedding_dim):
    model = Sequential(
        [
            Embedding(input_dim=vocab_size, output_dim=64),
            LSTM(32),
            Dense(1, activation="sigmoid"),
        ]
    )

    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

    model.summary()
    return model


def train_model(model, X_train, y_train, X_val, y_val, epochs=10, batch_size=64):
    """
    Trains the model using the provided data and saves the best version.
    """

    early_stopping = EarlyStopping(
        monitor="val_loss", patience=3, restore_best_weights=True, verbose=1
    )

    print("\n--- [Starting Model Training (V2)] ---")

    result = model.fit(
        X_train,
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_val, y_val),
        callbacks=[early_stopping],
        verbose=1,
    )

    model.save("sentiment_model_v3.keras")
    print("Model saved to sentiment_model_v3.keras")

    print("--- [Model Training Complete (V2)] ---")

    return result
