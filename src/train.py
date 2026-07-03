from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
# Load the Iris dataset
iris = load_iris()

X = iris.data
y = iris.target

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create the model
model = DecisionTreeClassifier(random_state=42)

# Train the model
model.fit(X_train, y_train)

# Make predictions
predictions = model.predict(X_test)

# Print the accuracy
print("Accuracy:", accuracy_score(y_test, predictions))

# Print the confusion matrix
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, predictions))
cm = confusion_matrix(y_test, predictions)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=iris.target_names
)

disp.plot()
plt.savefig("outputs/confusion_matrix.png")
plt.close()
# Print the classification report
print("\nClassification Report:")
print(classification_report(y_test, predictions, target_names=iris.target_names))

# Predict a new flower
new_flower = [[5.1, 3.5, 1.4, 0.2]]
prediction = model.predict(new_flower)

print("\nPredicted species:", iris.target_names[prediction][0])