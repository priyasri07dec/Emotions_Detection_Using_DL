import joblib

encoder = joblib.load("label_encoder.pkl")

print(encoder.classes_)