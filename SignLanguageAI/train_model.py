import pickle
from sklearn.neighbors import KNeighborsClassifier

with open("sign_data.pkl", "rb") as f:
    data, labels = pickle.load(f)

print(f"Training on {len(data)} samples with {len(set(labels))} unique labels...")

model = KNeighborsClassifier(n_neighbors=3)
model.fit(data, labels)

with open("knn_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("Model trained and saved as knn_model.pkl")