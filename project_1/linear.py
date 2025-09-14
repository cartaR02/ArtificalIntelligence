from sklearn import svm
from sklearn.preprocessing import StandardScaler
import numpy as np
import csv
import matplotlib.pyplot as plt

# Making dataset
X = []
Y = []
with open("groupA.csv", 'r', newline='') as infile:
    reader = csv.reader(infile)

    for row in reader:
        temp = [float(row[0]), float(row[1])]
        X.append(temp)
        Y.append(int(row[2]))


# Now lets train svm model

X = np.array(X)
Y = np.array(Y)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
model = svm.SVC(kernel='linear')
model.fit(X_scaled, Y)

# Lets predict for new input
n_data = np.array([[23615, 61370], [20000, 59000]])
n_data_scaled = scaler.transform(n_data)

pred = model.predict(n_data_scaled)
print(pred)

import matplotlib.pyplot as plt

# lets plot decision boundary for this
w = model.coef_[0]
b = model.intercept_[0] 

x_min, x_max = X_scaled[:, 0].min() - 1, X_scaled[:, 0].max() + 1
y_min, y_max = X_scaled[:, 1].min() - 1, X_scaled[:, 1].max() + 1

xx = np.linspace(x_min, x_max, 100)
yy = -(w[0] / w[1]) * xx - b / w[1]

plt.plot(xx, yy, 'k-', label="Decision boundary")

# Plot data points
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=Y, cmap=plt.cm.Set1, edgecolors="k")
plt.xlabel("Price (scaled)")
plt.ylabel("Weight (scaled)")
plt.legend()
plt.title("Linear SVM Decision Boundary")
plt.show()