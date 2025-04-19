import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Load dataset
df = pd.read_csv('bmi.csv')

# Encode categorical variables
gender_encoder = LabelEncoder()
df['Gender'] = gender_encoder.fit_transform(df['Gender'])

bmi_label_encoder = LabelEncoder()
df['BMIcase'] = bmi_label_encoder.fit_transform(df['BMIcase'])

# Features and labels
X = df[['Gender', 'Age', 'Height', 'Weight']]
y = df['BMIcase']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize models
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(n_estimators=100)
}

accuracies = {}
confusion_matrices = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    accuracies[name] = acc
    confusion_matrices[name] = confusion_matrix(y_test, preds)

    print(f"\n🔍 {name} Classification Report:")
    print(classification_report(y_test, preds, target_names=bmi_label_encoder.classes_))

# Save best model
best_model_name = max(accuracies, key=accuracies.get)
with open('bmi_model.pkl', 'wb') as file:
    pickle.dump(models[best_model_name], file)

with open('label_encoder.pkl', 'wb') as file:
    pickle.dump(bmi_label_encoder, file)

print(f"\n Best model '{best_model_name}' saved with accuracy: {accuracies[best_model_name]:.4f}")

#  Plot accuracy comparison
plt.figure(figsize=(8, 5))
sns.barplot(x=list(accuracies.keys()), y=list(accuracies.values()), palette="viridis")
plt.title("Model Accuracy Comparison")
plt.ylabel("Accuracy")
plt.ylim(0, 1)
plt.savefig("static/accuracy_comparison.png")
plt.show()

# 🔍 Plot confusion matrix for each model
for name, cm in confusion_matrices.items():
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=bmi_label_encoder.classes_,
                yticklabels=bmi_label_encoder.classes_)
    plt.title(f"{name} - Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.savefig(f"static/cm_{name.replace(' ', '_').lower()}.png")
    plt.show()
