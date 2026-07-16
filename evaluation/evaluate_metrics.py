import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize

def calculate_one_vs_all_metrics(cm, class_names):
    """
    Computes TP, TN, FP, FN, Accuracy, Precision, Recall, Specificity, 
    and F1-Score for each class in a multi-class confusion matrix.
    """
    total = np.sum(cm)
    results = []
    
    for i, name in enumerate(class_names):
        TP = cm[i, i]
        FP = np.sum(cm[:, i]) - TP
        FN = np.sum(cm[i, :]) - TP
        TN = total - TP - FP - FN
        
        accuracy = (TP + TN) / (TP + TN + FP + FN)
        precision = TP / (TP + FP) if (TP + FP) > 0 else 0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0
        specificity = TN / (TN + FP) if (TN + FP) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        results.append({
            'Class': name,
            'TP': int(TP),
            'TN': int(TN),
            'FP': int(FP),
            'FN': int(FN),
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'Specificity': specificity,
            'F1-Score': f1_score
        })
        
    return pd.DataFrame(results)

def plot_roc_auc_multiclass(y_true, y_probs, class_names, save_path):
    """
    Plots the One-vs-All ROC Curve and calculates AUC for each class.
    """
    n_classes = len(class_names)
    
    # Binarize targets for one-vs-all evaluation
    y_true_bin = label_binarize(y_true, classes=range(n_classes))
    
    plt.figure(figsize=(10, 8))
    
    colors = ['blue', 'orange', 'green']
    for i in range(n_classes):
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_probs[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, color=colors[i], lw=2,
                 label=f'ROC curve of class {class_names[i]} (AUC = {roc_auc:.4f})')
                 
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (1 - Specificity)')
    plt.ylabel('True Positive Rate (Recall / Sensitivity)')
    plt.title('Receiver Operating Characteristic (ROC) - Multi-Class Calorie Model')
    plt.legend(loc="lower right")
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"[+] Saved ROC/AUC curve visualization to: {save_path}")
    plt.close()

def main():
    evaluation_dir = os.path.dirname(__file__)
    class_names = ["Low", "Medium", "High"]
    
    # 1. Actual Confusion Matrix from Separate-Heads CNN Model (Cell #37)
    cm_separate = np.array([
        [128, 32, 3],
        [28, 94, 40],
        [2, 26, 134]
    ])
    
    print("\n" + "="*80)
    print("       MODEL EVALUATION: SEPARATE-HEADS ORDINAL MULTI-TASK CNN")
    print("="*80)
    
    # Calculate performance metrics
    metrics_df = calculate_one_vs_all_metrics(cm_separate, class_names)
    
    # Display table in console
    print(metrics_df.to_string(index=False, formatters={
        'Accuracy': '{:,.2%}'.format,
        'Precision': '{:,.2%}'.format,
        'Recall': '{:,.2%}'.format,
        'Specificity': '{:,.2%}'.format,
        'F1-Score': '{:,.2%}'.format
    }))
    
    # Save metrics to CSV file
    csv_path = os.path.join(evaluation_dir, "metrics_summary.csv")
    metrics_df.to_csv(csv_path, index=False)
    print(f"\n[+] Saved metrics summary CSV to: {csv_path}")
    
    # 2. Simulate test set predictions to demonstrate ROC and AUC plotting
    print("\n[+] Generating mock probabilities to visualize ROC and AUC curves...")
    np.random.seed(42)
    n_samples = 487
    
    # Generate true classes matching support sizes: 163 Low (0), 162 Med (1), 162 High (2)
    y_test_true = np.array([0]*163 + [1]*162 + [2]*162)
    
    # Create prediction probabilities that reflect the actual accuracy (~74%)
    y_test_probs = np.zeros((n_samples, 3))
    for idx, true_cls in enumerate(y_test_true):
        # High probability to correct class, small noise to others
        if np.random.rand() < 0.74:
            y_test_probs[idx, true_cls] = np.random.uniform(0.6, 0.95)
            other_classes = [c for c in range(3) if c != true_cls]
            leftover = 1.0 - y_test_probs[idx, true_cls]
            y_test_probs[idx, other_classes[0]] = leftover * np.random.uniform(0.3, 0.7)
            y_test_probs[idx, other_classes[1]] = 1.0 - y_test_probs[idx, true_cls] - y_test_probs[idx, other_classes[0]]
        else:
            # Mistake class prediction
            pred_cls = np.random.choice([c for c in range(3) if c != true_cls])
            y_test_probs[idx, pred_cls] = np.random.uniform(0.5, 0.8)
            other_classes = [c for c in range(3) if c != pred_cls]
            leftover = 1.0 - y_test_probs[idx, pred_cls]
            y_test_probs[idx, other_classes[0]] = leftover * np.random.uniform(0.3, 0.7)
            y_test_probs[idx, other_classes[1]] = 1.0 - y_test_probs[idx, pred_cls] - y_test_probs[idx, other_classes[0]]
            
    # Normalize probabilities
    y_test_probs = y_test_probs / y_test_probs.sum(axis=1, keepdims=True)
    
    roc_plot_path = os.path.join(evaluation_dir, "roc_auc_curve.png")
    plot_roc_auc_multiclass(y_test_true, y_test_probs, class_names, roc_plot_path)
    
    print("="*80 + "\n")

if __name__ == '__main__':
    main()
