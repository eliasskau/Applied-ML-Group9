import os
import numpy as np
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

#data laden en versimpelen

DATA_DIR = "data/images"
IMG_SIZE = 64
classes = ["alert", "angry", "frown", "happy", "relax"]

def process_image(image_path):
	image = Image.open(image_path).convert("RGB")
	image = image.resize((IMG_SIZE, IMG_SIZE))
	image_array = np.array(image)
	image_array = image_array/255.0
	image_flattend = image_array.flatten()
	return image_flattend

def load_dataset():
	X = []
	Y = []
	for label_index, class_name in enumerate(classes):
		class_folder = os.path.join(DATA_DIR, class_name)
		files = os.listdir(class_folder)
		print("Loading", class_name, "...")
		
		for filename in files:
			image_path = os.path.join(class_folder, filename)
			processed_image = process_image(image_path)
			X.append(processed_image)
			Y.append(label_index)
	X = np.array(X)
	Y = np.array(Y)
	return X, Y

X, Y = load_dataset()
print("X shape:", X.shape)
print("Y shape:", Y.shape)

#train test split

X_train, X_temp, Y_train, Y_temp = train_test_split( X, Y,  test_size = 0.30, random_state = 42, stratify = Y)
X_val, X_test, Y_val, Y_test = train_test_split(X_temp, Y_temp, test_size = 0.50, random_state=42, stratify = Y_temp)

print("X_train shape:", X_train.shape)
print("X_val shape:", X_val.shape)
print("X_test shape:", X_test.shape)

print("Y_train shape:", Y_train.shape)
print("Y_val shape:", Y_val.shape)
print("Y_test shape:", Y_test.shape)

#PCA toevoegen

pca = PCA(n_components= 100)
X_train_pca = pca.fit_transform(X_train)
X_val_pca = pca.transform(X_val)
X_test_pca = pca.transform(X_test)

print("X_train_pca shape:", X_train_pca.shape)
print("X_val_pca shape:", X_val_pca.shape)
print("X_test_pca shape:", X_test_pca.shape)


#model

model= LogisticRegression(max_iter = 1000)
model.fit(X_train_pca, Y_train)
print("Model Training finisched")

Y_val_pred = model.predict(X_val_pca)
val_accuracy = accuracy_score(Y_val, Y_val_pred)
val_f1 = f1_score(Y_val, Y_val_pred, average = "macro")
print("Validation accuracy:", val_accuracy)
print("Validation macro f1:", val_f1)
print(classification_report(Y_val, Y_val_pred, target_names=classes))
val_cm = confusion_matrix(Y_val, Y_val_pred)
print("Validation confision matrix:")
print(val_cm)
