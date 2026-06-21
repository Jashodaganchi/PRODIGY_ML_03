import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

data = []
labels = []
images = []

dataset_path = "PetImages"
categories = ["Cat", "Dog"]

# Load images
for label, category in enumerate(categories):
    folder_path = os.path.join(dataset_path, category)

    for file in os.listdir(folder_path)[:1500]:   # limit for faster training
        img_path = os.path.join(folder_path, file)

        img = cv2.imread(img_path)

        if img is None:
            continue

        img = cv2.resize(img, (128, 128))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray=cv2.equalizeHist(gray)  # Enhance contrast
        gray = gray / 255.0  # Normalize
        data.append(gray.flatten())
        labels.append(label)
        images.append(gray)

X = np.array(data)
y = np.array(labels)

# Split data
X_train, X_test, y_train, y_test, img_train, img_test = train_test_split(
    X, y, images, test_size=0.2, random_state=42
)

# Train model
model = SVC(kernel="rbf", C=1.0, gamma="scale")

print("Training model...")
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print("Accuracy:", accuracy)
print(report)

# Save output
with open("output.txt", "w") as file:
    file.write("Cat vs Dog Classification using SVM\n")
    file.write(f"Accuracy: {accuracy}\n")
    file.write("\nClassification Report:\n")
    file.write(report)

# Show sample predictions
plt.figure(figsize=(10, 5))

for i in range(4):
    plt.subplot(2, 2, i + 1)
    plt.imshow(img_test[i], cmap="gray")
    actual="Dog" if y_test[i] == 1 else "Cat"
    predicted="Dog" if y_pred[i] == 1 else "Cat"
    plt.title(f"Actual: {actual} Predicted: {predicted}")
    plt.axis("off")

plt.tight_layout()
plt.savefig("sample_predictions.png")
plt.show()

print("Results saved to output.txt")
print("Predictions saved to sample_predictions.png")