import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array, ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

def load_data_from_csv(csv_file):
    """Load images and labels from a CSV file."""
    df = pd.read_csv(csv_file)
    images = []
    labels = []

    # Map labels to class indices
    label_mapping = {'orange': 0, 'apple': 1}

    for index, row in df.iterrows():
        img_path = f'C:\\IERG4998\\image\\{row["filename"]}'  # Adjust path as necessary
        image = load_img(img_path, target_size=(224, 224))  # Resize to fit model input
        image = img_to_array(image) / 255.0  # Normalize image
        images.append(image)
        labels.append(label_mapping[row["label"]])  # Use mapped class index

    return np.array(images), np.array(labels)

def create_model(input_shape, num_classes):
    """Create a CNN model."""
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))  # Output layer
    return model

def train_model(csv_file):
    """Train the model using the data from the CSV file."""
    X, y = load_data_from_csv(csv_file)
    num_classes = len(np.unique(y))  # Should be 2 (0 and 1)

    # One-hot encode labels
    y = to_categorical(y, num_classes)  # Convert to one-hot encoding

    # Split data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    model = create_model((224, 224, 3), num_classes)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Fit the model with data augmentation and store the history
    datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    history = model.fit(datagen.flow(X_train, y_train, batch_size=32), 
                        validation_data=(X_val, y_val), 
                        epochs=10)

    model.save('fruit_classifier_model.keras')
    print("Model trained and saved successfully.")

    return history, model, X_val, y_val  # Return the history and validation data for plotting and evaluation

def plot_training_history(history):
    """Plot training and validation accuracy and loss."""
    # Plot training & validation accuracy values
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(loc='upper left')

    # Plot training & validation loss values
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(loc='upper left')

    plt.tight_layout()
    plt.show()

def evaluate_model(model, X_val, y_val):
    """Evaluate the model on the validation set."""
    loss, accuracy = model.evaluate(X_val, y_val)
    print(f'Validation Loss: {loss:.4f}')
    print(f'Validation Accuracy: {accuracy:.4f}')

# Example usage
if __name__ == "__main__":
    history, model, X_val, y_val = train_model('C:\\IERG4998\\UPLOADS\\train.csv')  # Replace with your actual CSV path
    plot_training_history(history)
    evaluate_model(model, X_val, y_val)