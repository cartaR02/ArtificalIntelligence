import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import numpy as np
import math
import os
import csv
from random import uniform
from sklearn.model_selection import train_test_split


def soft_unipolar(net):
    """Sigmoid activation [0,1]"""
    if net > 709:  # avoid overflow of exp
        return 1.0
    if net < -709:
        return 0.0
    return 1.0 / (1.0 + math.exp(-net))

def hard_unipolar(net, threshold=0):
    """Step function [0,1]"""
    return 1 if net > threshold else 0

def confusion_matrix(true, pred, name="", file_name=""):
    """Calculate confusion matrix for 0/1 labels"""
    tp = sum((t == 1 and p == 1) for t, p in zip(true, pred))
    tn = sum((t == 0 and p == 0) for t, p in zip(true, pred))
    fp = sum((t == 0 and p == 1) for t, p in zip(true, pred))
    fn = sum((t == 1 and p == 0) for t, p in zip(true, pred))
    
    total = tp + tn + fp + fn
    acc = (tp + tn) / total if total > 0 else 0.0
    
    print(f"\nConfusion Matrix ({file_name}, {name}):")
    print(f"           Predicted")
    print(f"           0    1")
    print(f"Actual 0  {tn:3}  {fp:3}")
    print(f"       1  {fn:3}  {tp:3}")
    print(f"Accuracy: {acc:.3f}")
    
    if tp + fn > 0:
        tpr = tp / (tp + fn)
        print(f"True Positive Rate: {tpr:.3f}")
    if tn + fp > 0:
        tnr = tn / (tn + fp)
        print(f"True Negative Rate: {tnr:.3f}")
    if fp + tn > 0:
        fpr = fp / (fp + tn)
        print(f"False Positive Rate: {fpr:.3f}")
    if fn + tp > 0:
        fnr = fn / (fn + tp)
        print(f"False Negative Rate: {fnr:.3f}")

    return {"tp": tp, "tn": tn, "fp": fp, "fn": fn, "acc": acc}

# Training parameters
max_epochs = 5000
alpha = 0.01  
gain = 0.5    # gain for soft activation

datasetDirectory = "datasets"
csvFiles = [os.path.join(datasetDirectory, file) for file in os.listdir(datasetDirectory) if file.endswith('.csv')]

# optional thresholds you had; used as epoch stopping threshold for soft (mean tanh error)
errorRates = [0.00001, 40, 700]

learning_rates = [0.1, 0.05, 0.01]  # A, B, C

# gain for soft activation
gain = 1.0

# train/test splits 
splits = [
    ("75/25", 0.75),
    ("25/75", 0.25)
]

for csvIterable, file in enumerate(csvFiles):
    print(f"\n{'='*60}")
    print(f"Training on file: {file}")
    print(f"{'='*60}\n")

    # Read raw feature columns (no bias) and labels
    raw_features = []
    Y = []
    with open(file, 'r', newline='') as infile:
        reader = csv.reader(infile)
        for row in reader:
            raw_features.append([float(row[0]), float(row[1])]) # price, weight
            Y.append(int(row[2]))  # 0/1 for unipolar

    raw_features = np.array(raw_features)
    Y = np.array(Y)
    n_samples = raw_features.shape[0]

    # Scale only the features (do NOT scale the bias)
    scaler = StandardScaler()
    scaled_feats = scaler.fit_transform(raw_features)

    # Build X with constant bias=1 column (NOT scaled)
    X = np.hstack([scaled_feats, np.ones((n_samples, 1))])
    npats, ni = X.shape

    print(f"Dataset: {npats} samples, {ni} features (including bias)")
    print(f"Class distribution: {np.sum(Y==0)} class 0, {np.sum(Y==1)} class 1")

    for split_name, train_size in splits:
        print(f"\n{'='*60}")
        print(f"Split: {split_name}")
        print(f"{'='*60}")
        
        # split the data
        indices = np.arange(npats)
        train_idx, test_idx = train_test_split(
            indices, train_size=train_size, random_state=42, stratify=Y
        )
        
        X_train = X[train_idx]
        X_test = X[test_idx]
        y_train = Y[train_idx]
        y_test = Y[test_idx]
        
        npats_train = len(X_train)
        npats_test = len(X_test)
        
        alpha = learning_rates[csvIterable]
        
        print(f"Training: {npats_train} samples, Testing: {npats_test} samples")
        print(f"Learning rate: {alpha}")

        
        # --- SOFT TRAINING ---
        print(f"\n--- Soft Unipolar Training ---")
        weights_soft = np.array([uniform(-0.5, 0.5) for _ in range(ni)])
        
        epochs_soft = 0
        for epoch in range(max_epochs):
            epochs_soft += 1
            total_error = 0
            
            for p in range(npats_train): 
                net = gain * np.dot(weights_soft, X_train[p]) 
                out = soft_unipolar(net)
                err = y_train[p] - out  
                total_error += err ** 2
                weights_soft += alpha * err * X_train[p] 
            
            # check stopping criterion
            if total_error < errorRates[csvIterable]:
                print(f"Converged at epoch {epochs_soft} with Total Error: {total_error:.6f}")
                break
        
        if epochs_soft == max_epochs:
            print(f"Max epochs reached. Final Total Error: {total_error:.6f}")
            
        # --- HARD UNIPOLAR TRAINING ---
        print(f"\n--- Hard Unipolar Training ---")
        weights_hard = np.array([uniform(-0.5, 0.5) for _ in range(ni)])
        
        epochs_hard = 0
        for epoch in range(max_epochs):
            epochs_hard += 1
            total_error = 0
            
            for p in range(npats_train): 
                net = np.dot(weights_hard, X_train[p]) 
                out = hard_unipolar(net)
                err = y_train[p] - out  
                total_error += err ** 2
                weights_hard += alpha * err * X_train[p] 
            
            # stop when perfect classification achieved
            if total_error == 0 or total_error < errorRates[csvIterable]:
                print(f"Converged at epoch {epochs_hard} with Total Error: {total_error:.6f}")
                break
        
        if epochs_hard == max_epochs:
            print(f"Max epochs reached. Final Total Error: {total_error:.6f}")

        # --- Evaluation (both) ---

        # evaluate soft on training set
        pred_soft_train = []
        for p in range(npats_train):
            net = gain * np.dot(weights_soft, X_train[p])
            out = soft_unipolar(net)
            pred = 1 if out > 0.5 else 0
            pred_soft_train.append(pred)

        print("\n[TRAINING SET]")
        confusion_matrix(y_train, pred_soft_train, "Soft Training", os.path.basename(file))

        # evaluate soft on testing set
        pred_soft_test = []
        for p in range(npats_test):
            net = gain * np.dot(weights_soft, X_test[p])
            out = soft_unipolar(net)
            pred = 1 if out > 0.5 else 0
            pred_soft_test.append(pred)

        print("\n[TESTING SET]")
        confusion_matrix(y_test, pred_soft_test, "Soft Testing", os.path.basename(file))
        
        # evaluate on training set
        pred_hard_train = []
        for p in range(npats_train):
            net = np.dot(weights_hard, X_train[p])
            out = hard_unipolar(net)
            pred = out 
            pred_hard_train.append(pred)

        print("\n[TRAINING SET]")
        confusion_matrix(y_train, pred_hard_train, "Hard Training", os.path.basename(file))

        # evaluate on testing set
        pred_hard_test = []
        for p in range(npats_test):
            net = np.dot(weights_hard, X_test[p])
            out = hard_unipolar(net)
            pred = out
            pred_hard_test.append(pred)

        print("\n[TESTING SET]")
        confusion_matrix(y_test, pred_hard_test, "Hard Testing", os.path.basename(file))
            
        # --- Plot Decision Boundaries ---
        
        # TRAINING DATA PLOTS
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Plot data points - Soft activation training
        ax1.scatter(X_train[y_train==0][:, 0], X_train[y_train==0][:, 1], 
                    c='red', marker='o', label='Class 0', s=50, edgecolors='k')
        ax1.scatter(X_train[y_train==1][:, 0], X_train[y_train==1][:, 1], 
                    c='blue', marker='s', label='Class 1', s=50, edgecolors='k')
        
        # Plot decision boundaries
        x_vals = np.linspace(X_train[:, 0].min()-0.5, X_train[:, 0].max()+0.5, 300)
        w0, w1, w2 = weights_soft
        
        # final soft boundary (emphasize)
        if abs(w1) > 1e-12:
            y_soft = -(w0/w1)*x_vals - (w2/w1)
            ax1.plot(x_vals, y_soft, 'g-', linewidth=2.5, label='Soft Unipolar')
        ax1.set_xlabel('Feature 1 (scaled)')
        ax1.set_ylabel('Feature 2 (scaled)')
        ax1.set_title(f'TRAINING - Soft - {split_name}')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot data points - Hard activation training
        ax2.scatter(X_train[y_train==0][:, 0], X_train[y_train==0][:, 1], 
                    c='red', marker='o', label='Class 0', s=50, edgecolors='k')
        ax2.scatter(X_train[y_train==1][:, 0], X_train[y_train==1][:, 1], 
                    c='blue', marker='s', label='Class 1', s=50, edgecolors='k')
        
        # final hard boundary
        w0, w1, w2 = weights_hard
        if abs(w1) > 1e-12:
            y_hard = -(w0 / w1) * x_vals - (w2 / w1)
            ax2.plot(x_vals, y_hard, 'purple', linestyle='--', linewidth=2.5, label='Hard Unipolar')
        ax2.set_xlabel('Feature 1 (scaled)')
        ax2.set_ylabel('Feature 2 (scaled)')
        ax2.set_title(f'TRAINING - Hard - {split_name}')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        # TESTING DATA PLOTS
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Plot data points - Soft activation testing
        ax1.scatter(X_test[y_test==0][:, 0], X_test[y_test==0][:, 1], 
                    c='red', marker='o', label='Class 0', s=50, edgecolors='k')
        ax1.scatter(X_test[y_test==1][:, 0], X_test[y_test==1][:, 1], 
                    c='blue', marker='s', label='Class 1', s=50, edgecolors='k')
        
        # Plot decision boundaries
        x_vals_test = np.linspace(X_test[:, 0].min()-0.5, X_test[:, 0].max()+0.5, 300)
        w0, w1, w2 = weights_soft
        
        # final soft boundary (emphasize)
        if abs(w1) > 1e-12:
            y_soft = -(w0/w1)*x_vals_test - (w2/w1)
            ax1.plot(x_vals_test, y_soft, 'g-', linewidth=2.5, label='Soft Unipolar')
        ax1.set_xlabel('Feature 1 (scaled)')
        ax1.set_ylabel('Feature 2 (scaled)')
        ax1.set_title(f'TESTING - Soft - {split_name}')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot data points - Hard activation testing
        ax2.scatter(X_test[y_test==0][:, 0], X_test[y_test==0][:, 1], 
                    c='red', marker='o', label='Class 0', s=50, edgecolors='k')
        ax2.scatter(X_test[y_test==1][:, 0], X_test[y_test==1][:, 1], 
                    c='blue', marker='s', label='Class 1', s=50, edgecolors='k')
        
        # final hard boundary
        w0, w1, w2 = weights_hard
        if abs(w1) > 1e-12:
            y_hard = -(w0 / w1) * x_vals_test - (w2 / w1)
            ax2.plot(x_vals_test, y_hard, 'purple', linestyle='--', linewidth=2.5, label='Hard Unipolar')
        ax2.set_xlabel('Feature 1 (scaled)')
        ax2.set_ylabel('Feature 2 (scaled)')
        ax2.set_title(f'TESTING - Hard - {split_name}')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()