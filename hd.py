import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import cv2
import numpy as np
from skimage.feature import hog
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import seaborn as sns
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk

# Function to extract HOG features from an image
def extract_hog_features(image):
    resized_image = cv2.resize(image, (80, 80))  # Resize image to a fixed size
    fd = hog(resized_image, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=False, channel_axis=-1)
    return fd

# Function to load images from directory
def load_images_from_dir(directory):
    images = []
    labels = []
    
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.jpg') or file.endswith('.png'):
                filepath = os.path.join(root, file)
                label = os.path.basename(root)
                labels.append(label)
                image = cv2.imread(filepath)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB (OpenCV uses BGR by default)
                images.append(image)
    
    return images, labels

# def create_cnn_model(input_shape, num_classes):
#     model = models.Sequential()
#     model.add(layers.Dense(128, activation='relu', input_shape=input_shape))
#     model.add(layers.Dense(num_classes, activation='softmax'))
#     return model

# Main function to train and evaluate the CNN model
def train_evaluate_cnn_model(dataset_dir):
    # Step 1: Load images and labels from the specified directory
    images, labels = load_images_from_dir(dataset_dir)

    # Step 2: Preprocess images and extract features (HOG)
    hog_features = []
    for image in images:
        hog_feature = extract_hog_features(image)
        hog_features.append(hog_feature)
    
    X = np.array(hog_features)
    y = np.array(labels)

    # Step 3: Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Step 4: Encode labels
    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)
    y_test_encoded = label_encoder.transform(y_test)
    num_classes = len(label_encoder.classes_)

    cnn_model = models.Sequential([
        layers.Reshape((54, 54, 1), input_shape=(2916,)),
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),

        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(num_classes, activation='softmax')
    ])

    cnn_model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    history = cnn_model.fit(X_train, y_train_encoded, epochs=8, batch_size=32, validation_data=(X_test, y_test_encoded))

    # Step 6: Evaluate the model
    test_loss, test_acc = cnn_model.evaluate(X_test, y_test_encoded)
    print(f'Test accuracy: {test_acc}')

    # Step 7: Save the model
    cnn_model.save('hand_gesture_model.h5', save_format='h5')
    print('Model saved as hand_gesture_model.h5')

    # Plot the training and validation loss and accuracy
    plot_training_history(history)

    # Step 8: Compute confusion matrix
    # y_pred = cnn_model.predict(X_test)
    # y_pred_classes = np.argmax(y_pred, axis=1)
    
    
 
    return cnn_model, label_encoder

# Function to plot the training history
def plot_training_history(history):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(len(acc))

    plt.figure(figsize=(12, 8))

    plt.subplot(2, 1, 1)
    plt.plot(epochs, acc, 'r', label='Training accuracy')  # Red for training accuracy
    plt.plot(epochs, val_acc, 'b', label='Validation accuracy')  # Blue for validation accuracy
    plt.title('Training and validation accuracy')
    
    plt.ylabel('Accuracy')
    plt.legend()

    plt.subplot(2, 1, 2)  # Changed subplot index to 2 for the second plot
    plt.plot(epochs, loss, 'r', label='Training loss')  # Red for training loss
    plt.plot(epochs, val_loss, 'b', label='Validation loss')  # Blue for validation loss
    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.show()

# Function to classify an image using the trained model
def classify_image(model, label_encoder, image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    hog_feature = extract_hog_features(image)
    hog_feature = np.expand_dims(hog_feature, axis=0)
     
    predictions = model.predict(hog_feature)
    predicted_class_idx = np.argmax(predictions)
    predicted_class = label_encoder.inverse_transform([predicted_class_idx])[0]
    return predicted_class

# Function to handle the classification button click
def classify_button_click():
    file_path = filedialog.askopenfilename()
    if not file_path:
        return

    # Display the selected image
    image = Image.open(file_path)
    image = image.resize((200, 200), Image.Resampling.LANCZOS)  # Updated to LANCZOS
    img = ImageTk.PhotoImage(image)
    panel.configure(image=img)
    panel.image = img

    # Get the actual class from the file path
    actual_class = os.path.splitext(os.path.basename(file_path))[0]

    # Predict the class of the image
    predicted_class = classify_image(model, label_encoder, file_path)

    # Display the prediction results
    result_text.set(f"Predicted Class: {predicted_class}\nActual Class: {actual_class}")
    messagebox.showinfo("Prediction", f"Predicted Class: {predicted_class}\nActual Class: {actual_class}")

if __name__ == "__main__":
    dataset_dir = 'C:/Users/Dell/Downloads/hs/hs/Static_ISL'  # Specify your dataset directory
    model, label_encoder = train_evaluate_cnn_model(dataset_dir)
    
    # Create the main window
    root = tk.Tk()
    root.title("Hand Gesture Classifier")

    # Create a button to classify an image
    classify_button = tk.Button(root, text="Classify Image", command=classify_button_click)
    classify_button.pack(pady=20)

    # Create a label to display the image
    panel = tk.Label(root)
    panel.pack(pady=20)

    # Create a label to display the result
    result_text = tk.StringVar()
    result_label = tk.Label(root, textvariable=result_text)
    result_label.pack(pady=20)

    # Start the Tkinter event loop
    root.mainloop()
