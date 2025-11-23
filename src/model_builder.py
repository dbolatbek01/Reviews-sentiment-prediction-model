import tensorflow
from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional
from keras.metrics import Precision, Recall
from keras.callbacks import EarlyStopping


def create_model(vocab_size, max_len, embedding_dim):
    model = Sequential(
        [
            Embedding(
                input_dim=vocab_size, output_dim=embedding_dim, input_length=max_len
            ),
            Bidirectional(LSTM(64, return_sequences=True)),
            Dropout(0.5),
            Bidirectional(LSTM(32)),
            Dropout(0.3),
            Dense(3, activation="softmax"),
        ]
    )

    model.compile(
        loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"]
    )

    model.summary()
    return model


def train_model(model, X_train, y_train, X_val, y_val, epochs=10, batch_size=64):
    early_stopping = EarlyStopping(
        monitor="val_loss", patience=3, restore_best_weights=True, verbose=1
    )

    print("\n--- [Starting Model Training] ---")

    result = model.fit(
        X_train,
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_val, y_val),
        callbacks=[early_stopping],
        verbose=1,
    )

    model.save("sentiment_model.keras")
    print("Model saved to sentiment_model.keras")

    print("--- [Model Training Complete] ---")

    return result
