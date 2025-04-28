# Farm-Intrusion-Detection
A system that  uses vision based object  detection to  detect  unauthorized  entries into  agricultural  fields.  To detect  unauthorized  entries  (humans or  animals) into  agricultural  fields for better  farm security. 


Code for Project 

import os
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
from google.colab import drive
drive.mount('/content/drive')
import zipfile
import os

zip_path = "/content/drive/MyDrive/archive.zip"
extract_path = "/content/archive"

with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(extract_path)

base_path = "/content/archive"


IMG_SIZE = (224, 224)
BATCH_SIZE = 32

# Step 3: Use validation_split to split train and validation sets
datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2  # 20% for validation
)
train_data = datagen.flow_from_directory(
    base_path,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training',
    shuffle=True
)
val_data = datagen.flow_from_directory(
    base_path,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation',
    shuffle=True
)
class_counts = [len(os.listdir(os.path.join(base_path, class_name))) for class_name in os.listdir(base_path)]
class_names = os.listdir(base_path)

plt.figure(figsize=(14, 7))
plt.bar(class_names, class_counts)
plt.xticks(rotation=90)
plt.xlabel('Classes (Intrusion Types)')
plt.ylabel('Number of Images')
plt.title('Class Distribution in Dataset')
plt.show()
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
    MaxPooling2D(2, 2),

    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),

    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),

    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(train_data.num_classes, activation='softmax')
])
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Step 6: Train model
EPOCHS = 10
history = model.fit(
    train_data,
    validation_data=val_data ,
    epochs=EPOCHS
)
model.save("intrusion_detection_model.h5")
test_loss, test_accuracy = model.evaluate(test_data)
print(f"Test Accuracy: {test_accuracy:.2f}")
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()
def test_sample_images(num_samples=5):
    print("\n Sample Predictions from Validation Set:")
    sample_batch = next(val_data)
    sample_images, sample_labels = sample_batch[0][:num_samples], sample_batch[1][:num_samples]
    predictions = model.predict(sample_images)

    for i in range(num_samples):
        plt.imshow(sample_images[i])
        plt.axis('off')
        pred_label = class_names[np.argmax(predictions[i])]
        true_label = class_names[np.argmax(sample_labels[i])]
        plt.title(f"Pred: {pred_label}\nTrue: {true_label}")
        plt.show()

test_sample_images(num_samples=5)
