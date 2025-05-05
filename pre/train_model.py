import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

data = pd.read_csv('../data/gesture_data.csv')
X = data[['thumb_x', 'thumb_y', 'index_x', 'index_y']]
y = data['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

models = {
    'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=3),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'Support Vector Machine': SVC(kernel='linear', random_state=42)
}

best_score = 0
best_model = None
best_model_name = ''

print("Model Comparison Results:")
print("=" * 40)
for name, model in models.items():
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred) * 100
    
    print(f"{name}: {accuracy:.2f}%")
    
    if accuracy > best_score:
        best_score = accuracy
        best_model = model
        best_model_name = name

print("=" * 40)
print(f"\nBest model: {best_model_name} with accuracy: {best_score:.2f}%")

output_filename = 'best_gesture_model.pkl'
joblib.dump(best_model, output_filename)
print(f"\nBest model saved as '{output_filename}'")