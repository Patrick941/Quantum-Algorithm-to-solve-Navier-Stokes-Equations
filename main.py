import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import Callback
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
import os

# Load CIFAR-10 dataset
(X_train, y_train), (X_test, y_test) = tf.keras.datasets.cifar10.load_data()

# Normalize the data
X_train, X_test = X_train / 255.0, X_test / 255.0

# Build a more complex neural network model
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Custom callback to stop training when divergence exceeds threshold
class DivergenceCallback(Callback):
    def __init__(self, threshold=0.05):
        super(DivergenceCallback, self).__init__()
        self.threshold = threshold

    def on_epoch_end(self, epoch, logs=None):
        if epoch < 10:
            return
        acc = logs.get('accuracy')
        val_acc = logs.get('val_accuracy')
        if abs(acc - val_acc) > self.threshold:
            print(f"\nStopping training at epoch {epoch + 1} due to divergence threshold.")
            self.model.stop_training = True

# Train the model with the custom callback
history = model.fit(X_train, y_train, epochs=50, batch_size=64, validation_split=0.2, callbacks=[DivergenceCallback()])

# Evaluate the model on the test data
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f'Test accuracy: {test_acc}')

# Create Images directory if it doesn't exist
os.makedirs('Images', exist_ok=True)

# Plot training & validation accuracy values
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.savefig('Images/model_accuracy.png')

# Plot training & validation loss values
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.savefig('Images/model_loss.png')


# Show a few sample images with their predicted labels
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

plt.figure(figsize=(10, 10))
for i in range(9):
    plt.subplot(3, 3, i + 1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    img = X_test[i]
    plt.imshow(img, cmap=plt.cm.binary)
    predicted_label = np.argmax(model.predict(np.expand_dims(img, axis=0)))
    true_label = y_test[i][0]
    color = 'blue' if predicted_label == true_label else 'red'
    plt.xlabel(f"{class_names[predicted_label]} ({class_names[true_label]})", color=color)
plt.savefig('Images/sample_predictions.png')