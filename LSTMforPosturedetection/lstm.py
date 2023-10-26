# import numpy as np
# import tensorflow as tf
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import LSTM, Dense
# import pandas as pd
# import matplotlib.pyplot as plt
# from tensorflow.keras.callbacks import EarlyStopping
# from tensorflow.keras.layers import GRU


# early_stop = EarlyStopping(monitor='val_loss', patience=20, verbose=1, restore_best_weights=True)

# def build_and_train_model(activation_function):
#     # 2. Build the LSTM model
#     model = Sequential()
#     model.add(GRU(50, input_shape=(X_train.shape[1], X_train.shape[2]), activation=activation_function))
#     model.add(Dense(4, activation='softmax'))  # 4 classes (excluding unknown)
#     model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
#     # Print model summary
#     print(f"Model Summary for {activation_function} Activation:")
#     model.summary()    

#     # 3. Train the model
#     history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_val, y_val), shuffle=True, callbacks=[early_stop])


#     # 4. Predict with the model
#     predictions = model.predict(X_test)

#     # Classify as 'unknown' if confidence is below threshold
#     threshold = 0.6
#     predicted_class = np.argmax(predictions, axis=1)
#     below_threshold = predictions.max(axis=1) < threshold
#     predicted_class[below_threshold] = 5  # 5 represents the "unknown" class

#     # Shift classes back from 0-3 to 1-4
#     predicted_class = predicted_class + 1
#     y_test_shifted = y_test + 1

#     # Save the trained model
#     model_name = f"gru_model_{activation_function}.h5"
#     model.save(model_name)

#     # Check the accuracy (without considering 'unknown' as errors)
#     accuracy = np.mean((y_test_shifted == predicted_class) | (below_threshold))
#     print(f"{activation_function} Activation Accuracy: {accuracy * 100:.2f}%")
#     return accuracy, history

# # 1. Dataset preparation
# train_data = pd.read_csv("train_data.csv").values
# test_data = pd.read_csv("test_data.csv").values
# val_data = pd.read_csv("validate_data.csv").values

# X_train, y_train = train_data[:, :-1], train_data[:, -1]
# X_test, y_test = test_data[:, :-1], test_data[:, -1]
# X_val, y_val = val_data[:, :-1], val_data[:, -1]

# # Adjust labels from 1-4 to 0-3
# y_train = y_train - 1
# y_val = y_val - 1
# y_test = y_test - 1

# X_train = X_train.reshape(X_train.shape[0], 1, X_train.shape[1])
# X_test = X_test.reshape(X_test.shape[0], 1, X_test.shape[1])
# X_val = X_val.reshape(X_val.shape[0], 1, X_val.shape[1])

# activation_functions = ['sigmoid', 'tanh', 'relu']
# results = {}

# def plot_history(history, activation):
#     # Plot training & validation accuracy values
#     plt.figure(figsize=(12, 5))
    
#     plt.subplot(1, 2, 1)
#     plt.plot(history.history['accuracy'])
#     plt.plot(history.history['val_accuracy'])
#     plt.title(f'{activation} Activation Accuracy')
#     plt.ylabel('Accuracy')
#     plt.xlabel('Epoch')
#     plt.legend(['Train', 'Validation'], loc='upper left')
    
#     # Plot training & validation loss values
#     plt.subplot(1, 2, 2)
#     plt.plot(history.history['loss'])
#     plt.plot(history.history['val_loss'])
#     plt.title(f'{activation} Activation Loss')
#     plt.ylabel('Loss')
#     plt.xlabel('Epoch')
#     plt.legend(['Train', 'Validation'], loc='upper left')
    
#     plt.tight_layout()
#     plt.show()


# for activation in activation_functions:
#     accuracy, history = build_and_train_model(activation)
#     results[activation] = accuracy

#     # Print training, validation accuracy for the last epoch
#     print(f"{activation} Activation - Training Accuracy: {history.history['accuracy'][-1] * 100:.2f}%")
#     print(f"{activation} Activation - Validation Accuracy: {history.history['val_accuracy'][-1] * 100:.2f}%")

#     plot_history(history, activation)

# print("\nFinal Results:")
# for activation, accuracy in results.items():
#     print(f"{activation} Activation - General Accuracy: {accuracy * 100:.2f}%")
#     print(f"{activation} Activation - Training Accuracy: {history.history['accuracy'][-1] * 100:.2f}%")
#     print(f"{activation} Activation - Validation Accuracy: {history.history['val_accuracy'][-1] * 100:.2f}%")
#     print("-----")

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import EarlyStopping

early_stop = EarlyStopping(monitor='val_loss', patience=20, verbose=1, restore_best_weights=True)

def build_and_train_simple_model():
    # Build the Simple Dense model
    model = Sequential()
    model.add(Flatten(input_shape=(X_train.shape[1], X_train.shape[2])))  # Flatten the input
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(4, activation='softmax'))  # 4 classes
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    # Print model summary
    print("Model Summary for Simple Feed-forward Network:")
    model.summary()    

    # Train the model
    history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_val, y_val), shuffle=True, callbacks=[early_stop])

    # Predict with the model
    predictions = model.predict(X_test)

    # Classify as 'unknown' if confidence is below threshold
    threshold = 0.6
    predicted_class = np.argmax(predictions, axis=1)
    below_threshold = predictions.max(axis=1) < threshold
    predicted_class[below_threshold] = 5  # 5 represents the "unknown" class

    # Shift classes back from 0-3 to 1-4
    predicted_class = predicted_class + 1
    y_test_shifted = y_test + 1

    # Save the trained model
    model_name = "dense_model.h5"
    model.save(model_name)

    # Check the accuracy (without considering 'unknown' as errors)
    accuracy = np.mean((y_test_shifted == predicted_class) | (below_threshold))
    print(f"Simple Feed-forward Network Accuracy: {accuracy * 100:.2f}%")
    return accuracy, history

# Dataset preparation remains unchanged
train_data = pd.read_csv("train_data.csv").values
test_data = pd.read_csv("test_data.csv").values
val_data = pd.read_csv("validate_data.csv").values

X_train, y_train = train_data[:, :-1], train_data[:, -1]
X_test, y_test = test_data[:, :-1], test_data[:, -1]
X_val, y_val = val_data[:, :-1], val_data[:, -1]

# Adjust labels from 1-4 to 0-3
y_train = y_train - 1
y_val = y_val - 1
y_test = y_test - 1

X_train = X_train.reshape(X_train.shape[0], 1, X_train.shape[1])
X_test = X_test.reshape(X_test.shape[0], 1, X_test.shape[1])
X_val = X_val.reshape(X_val.shape[0], 1, X_val.shape[1])

accuracy, history = build_and_train_simple_model()

# Your plotting code remains unchanged
def plot_history(history):
    # Plot training & validation accuracy values
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Feed-forward Network Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    
    # Plot training & validation loss values
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Feed-forward Network Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    
    plt.tight_layout()
    plt.show()


plot_history(history)

# Print the results
print("\nFinal Results:")
print(f"Feed-forward Network - General Accuracy: {accuracy * 100:.2f}%")
print(f"Feed-forward Network - Training Accuracy: {history.history['accuracy'][-1] * 100:.2f}%")
print(f"Feed-forward Network - Validation Accuracy: {history.history['val_accuracy'][-1] * 100:.2f}%")
