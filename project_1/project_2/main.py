import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import numpy as np
import math
import os
import csv
from random import uniform

def fbip(net, k=0.5):
    """ bipolar sigmoid (tanh) activation, overflow-safe """
    val = -2 * k * net
    if val > 709:    # avoid overflow of exp
        return -1.0
    if val < -709:
        return 1.0
    return 2.0 / (1.0 + math.exp(val)) - 1.0

def fHard(net, threshold=0):
    """ hard activation """
    return 1 if net > threshold else -1

def confusion_matrix(true, pred, name="", file_name=""):
    tp = sum((t == 1 and p == 1) for t, p in zip(true, pred))
    tn = sum((t == -1 and p == -1) for t, p in zip(true, pred))
    fp = sum((t == -1 and p == 1) for t, p in zip(true, pred))
    fn = sum((t == 1 and p == -1) for t, p in zip(true, pred))
    acc = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0
    print(f"\nConfusion Matrix ({file_name} , {name}):")
    print(f"TP={tp}, FP={fp}")
    print(f"FN={fn}, TN={tn}")
    print(f"Accuracy: {acc:.3f}")

# Training parameters
max_epochs = 5000
alpha = 0.1
k = 0.5        # gain for soft activation

datasetDirectory = "datasets"
csvFiles = [os.path.join(datasetDirectory, file) for file in os.listdir(datasetDirectory) if file.endswith('.csv')]

# optional thresholds you had; used as epoch stopping threshold for soft (mean tanh error)
errorRates = [40, 700, 0.000001]

csvIterable = 0
for file in csvFiles:
    print(f"\n=== Training on file: {file} ===\n")

    # Read raw feature columns (no bias) and labels
    raw_features = []
    Y = []
    with open(file, 'r', newline='') as infile:
        reader = csv.reader(infile)
        for row in reader:
            raw_features.append([float(row[0]), float(row[1])])  # price, weight
            Y.append(1 if int(row[2]) == 1 else -1)

    raw_features = np.array(raw_features)
    Y = np.array(Y)
    n_samples = raw_features.shape[0]

    # Scale only the features (do NOT scale the bias)
    scaler = StandardScaler()
    scaled_feats = scaler.fit_transform(raw_features)

    # Build X with constant bias=1 column (NOT scaled)
    X = np.hstack([scaled_feats, np.ones((n_samples, 1), dtype=float)])  # shape (n, 3)
    npats, ni = X.shape

    # --- SOFT TRAINING ---
    weights_soft = np.array([uniform(-0.5, 0.5) for _ in range(ni)])
    weight_history_soft = [weights_soft.copy()]

    epochs_soft = 0
    for epoch in range(max_epochs):
        epochs_soft += 1
        for p in range(npats):
            net = np.dot(weights_soft, X[p])
            out = fbip(net, k)
            err = Y[p] - out
            learn = alpha * err
            weights_soft = weights_soft + learn * X[p]

        weight_history_soft.append(weights_soft.copy())

        # compute mean absolute tanh error this epoch (aggregate measure)
        epoch_err = np.mean([abs(Y[p] - fbip(np.dot(weights_soft, X[p]), k)) for p in range(npats)])
        # use provided errorRates threshold if it makes sense, otherwise don't stop too early
        threshold = errorRates[csvIterable] if csvIterable < len(errorRates) else 0.0
        if epoch_err < threshold:
            break

    print(f"Soft training finished after {epochs_soft} epochs. Final mean tanh error: {epoch_err:.6f}")

    # --- HARD TRAINING (Perceptron rule with step activation) ---
    weights_hard = np.array([uniform(-0.5, 0.5) for _ in range(ni)])
    weight_history_hard = [weights_hard.copy()]

    epochs_hard = 0
    for epoch in range(max_epochs):
        epochs_hard += 1
        for p in range(npats):
            net = np.dot(weights_hard, X[p])
            out = fHard(net)
            err = Y[p] - out
            learn = alpha * err
            weights_hard = weights_hard + learn * X[p]

        weight_history_hard.append(weights_hard.copy())

        # stop when perfect classification achieved
        preds = [fHard(np.dot(weights_hard, X[p])) for p in range(npats)]
        mis = sum(1 for t, pr in zip(Y, preds) if t != pr)
        if mis == 0:
            break

    print(f"Hard training finished after {epochs_hard} epochs. Misclassifications: {mis}")

    # --- Evaluation (both) ---
    all_pred_soft_train = []
    for p in range(npats):
        net = np.dot(weights_soft, X[p])
        pred = 1 if fbip(net, k) > 0 else -1
        all_pred_soft_train.append(pred)
    confusion_matrix(Y, all_pred_soft_train, "Soft Training", os.path.basename(file))

    all_pred_hard_train = []
    for p in range(npats):
        net = np.dot(weights_hard, X[p])
        pred = fHard(net)
        all_pred_hard_train.append(pred)
    confusion_matrix(Y, all_pred_hard_train, "Hard Training", os.path.basename(file))

    # --- Plot decision boundaries ---
    plt.figure(figsize=(7,7))
    plt.axhline(0, color="black", linewidth=0.5)
    plt.axvline(0, color="black", linewidth=0.5)

    # plot training points (scaled features)
    for p in range(npats):
        plt.scatter(X[p][0], X[p][1],
                    color="red" if Y[p] == -1 else "blue",
                    label="d=-1" if (Y[p] == -1 and p == 0) else "d=+1" if (Y[p] == 1 and p == 0) else "")

    # prepare x range in scaled feature space
    pad = 0.2
    x_min, x_max = X[:,0].min(), X[:,0].max()
    y_min, y_max = X[:,1].min(), X[:,1].max()
    x_vals = np.linspace(x_min - pad, x_max + pad, 200)

    # Soft boundaries: sample up to N lines for clarity (avoid plotting thousands)
    n_history = len(weight_history_soft)
    max_lines = 60
    if n_history <= max_lines:
        indices = list(range(n_history))
    else:
        indices = np.linspace(0, n_history - 1, max_lines, dtype=int)

    colors = plt.cm.plasma(np.linspace(0, 1, len(indices)))
    for idx, col in zip(indices, colors):
        w = weight_history_soft[idx]
        w0, w1, w2 = w  # w0* x + w1* y + w2*1 = 0
        if abs(w1) > 1e-12:
            y_vals = -(w0 / w1) * x_vals - (w2 / w1)
        else:
            y_vals = np.full_like(x_vals, -w2 / w0 if abs(w0) > 1e-12 else y_min)
        alpha_line = 0.5 if idx != indices[-1] else 0.95
        lw = 1.0 if idx != indices[-1] else 2.0
        plt.plot(x_vals, y_vals, color=col, alpha=alpha_line, linewidth=lw,
                 label="Soft boundary (final)" if idx == indices[-1] else None)

    # final soft boundary (emphasize)
    final_soft = weight_history_soft[-1]
    w0, w1, w2 = final_soft
    if abs(w1) > 1e-12:
        y_vals = -(w0 / w1) * x_vals - (w2 / w1)
    else:
        y_vals = np.full_like(x_vals, -w2 / w0 if abs(w0) > 1e-12 else y_min)
    plt.plot(x_vals, y_vals, color="red", linestyle='-', linewidth=2.5, label="Soft final")

    # final hard boundary
    final_hard = weight_history_hard[-1]
    w0, w1, w2 = final_hard
    if abs(w1) > 1e-12:
        y_vals = -(w0 / w1) * x_vals - (w2 / w1)
    else:
        y_vals = np.full_like(x_vals, -w2 / w0 if abs(w0) > 1e-12 else y_min)
    plt.plot(x_vals, y_vals, color="green", linestyle="--", linewidth=2.5, label="Hard final")

    plt.xlim(x_min - pad, x_max + pad)
    plt.ylim(y_min - pad, y_max + pad)
    plt.title(f"Soft vs Hard Perceptron Training ({os.path.basename(file)})")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper_left' if hasattr(plt, 'upper_left') else 'upper left')
    plt.show()

    csvIterable += 1
