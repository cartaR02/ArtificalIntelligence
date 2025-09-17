from sklearn import svm
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler
import numpy as np
import csv
import matplotlib.pyplot as plt
import os

# grab all files to run at once
datasetDirectory = "datasets"
csvFiles = [os.path.join(datasetDirectory, file) for file in os.listdir(datasetDirectory) if file.endswith('.csv')]

# storage to calc confusion matrix
all_true_labels = []
all_predictions = []

# iterate through all files.
for file in csvFiles:
    X = []
    Y = []

    with open(file, 'r', newline='') as infile:
        reader = csv.reader(infile)
        for row in reader:
            temp = [float(row[0]), float(row[1])]
            X.append(temp)
            Y.append(int(row[2]))

    X = np.array(X)
    Y = np.array(Y)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    model = svm.SVC(kernel='linear')
    model.fit(X_scaled, Y)

    pred = model.predict(X_scaled)

    # Predict based on unseen points
    n_data = np.array([[23615, 61370], [20000, 59000]])
    n_data_scaled = scaler.transform(n_data)
    n_pred = model.predict(n_data_scaled)

    print(f"Predictions for {n_data.tolist()} -> {n_pred.tolist()}")

    all_true_labels.extend(Y)
    all_predictions.extend(pred)

    plt.figure()

    w = model.coef_[0]
    b = model.intercept_[0]

    x_min, x_max = X_scaled[:, 0].min() - 1, X_scaled[:, 0].max() + 1
    y_min, y_max = X_scaled[:, 1].min() - 1, X_scaled[:, 1].max() + 1

    xx = np.linspace(x_min, x_max, 100)

    if w[1] != 0:
        yy = -(w[0] / w[1]) * xx - b / w[1]
        plt.plot(xx, yy, 'k-', label="Decision boundary")

    plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=Y, cmap=plt.cm.Set1, edgecolors="k", marker="o")
    plt.xlabel("Feature 1 (scaled)")
    plt.ylabel("Feature 2 (scaled)")
    plt.legend()
    plt.title(f"Linear SVM Decision Boundary for {os.path.basename(file)}")

    plt.show(block=False)

cm = confusion_matrix(all_true_labels, all_predictions)
tn, fp, fn, tp = cm.ravel()

accuracy = accuracy_score(all_true_labels, all_predictions)
error_rate = 1 - accuracy
true_positive_rate = tp / (tp + fn)
true_negative_rate = tn / (tn + fp)
false_positive_rate = fp / (fp + tn)
false_negative_rate = fn / (fn + tp)

print("Confusion Matrix:")
print(cm)
print(f"Accuracy: {accuracy:.2f}")
print(f"Error Rate: {error_rate:.2f}")
print(f"True Positive Rate (Sensitivity): {true_positive_rate:.2f}")
print(f"True Negative Rate (Specificity): {true_negative_rate:.2f}")
print(f"False Positive Rate: {false_positive_rate:.2f}")
print(f"False Negative Rate: {false_negative_rate:.2f}")

plt.show()
