from keras.layers import Input, Dense
from keras.models import Model
from keras import regularizers
from sklearn.model_selection import train_test_split
import pandas as pd


def encode(X, encoding_dim=2, prefix=''):
    input_img = Input(shape=(X.shape[1],))
    encoded = Dense(units=128, activation='relu')(input_img)
    encoded = Dense(units=encoding_dim, activation='relu')(encoded)
    decoded = Dense(units=128, activation='relu')(encoded)
    decoded = Dense(units=X.shape[1], activation='sigmoid')(decoded)
    autoencoder = Model(input_img, decoded)
    encoder = Model(input_img, encoded)
    encoded_input = Input(shape=(encoding_dim,))
    #decoder_layer = autoencoder.layers[-1]
    #decoder = Model(encoded_input, decoder_layer(encoded_input))
    autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
    X_train, X_test = train_test_split(X, random_state=42)
    autoencoder.fit(X_train, X_train,
                    epochs=50,
                    batch_size=256,
                    shuffle=True,
                    validation_data=(X_test, X_test),
                    verbose=2)
    encoded = encoder.predict(X)
    columns = [f'{prefix}autoencoder_{i}' for i in range(encoding_dim)]
    df = pd.DataFrame(encoded, columns=columns)
    return df