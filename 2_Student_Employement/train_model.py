import pandas as pd
import pickle
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

df = pd.read_csv("Student-Employability-Datasets.csv") 

df = df.iloc[:, 1:]

X = df.iloc[:, :-1] 
y = df.iloc[:, -1]  

y = y.map({"Employable": 1, "LessEmployable": 0})

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

model = MLPClassifier(hidden_layer_sizes=(10,), activation='relu', max_iter=500, random_state=42)
model.fit(X_train, y_train)

with open("model.pkl", "wb") as model_file:
    pickle.dump(model, model_file)

with open("scaler.pkl", "wb") as scaler_file:
    pickle.dump(scaler, scaler_file)

print("Model and scaler saved successfully!")
