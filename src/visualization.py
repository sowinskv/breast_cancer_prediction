import matplotlib
matplotlib.use('Agg') # niestety musze generowac bez gui
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import precision_recall_curve, average_precision_score


def plot_precision_recall_curve(y_test, y_prob, output_dir):
    if y_prob is None:
        print("model nie obsługuje predykcji prawdopodobieństwa.")
        return

    precision, recall, _ = precision_recall_curve(y_test, y_prob)
    avg_precision = average_precision_score(y_test, y_prob)

    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, label=f'precision-recall curve (AP = {avg_precision:.2f})')
    plt.xlabel('recall')
    plt.ylabel('precyzja')
    plt.title('precision-recall curve')
    plt.legend(loc="lower left")
    plt.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, "precision_recall_curve.png"))
    plt.close()


def plot_confusion_matrix(cm, output_dir):
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('przewidywane')
    plt.ylabel('prawdziwe')
    plt.title('macierz konfuzji')
    plt.savefig(f"{output_dir}/confusion_matrix.png")
    plt.close()


def plot_roc_curve(y_test, y_prob, output_dir):
    if y_prob is None:
        print("model nie obsługuje predykcji prawdopodobieństwa.")
        return

    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--', label='losowa klasyfikacja')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('krzywa ROC-AUC')
    plt.legend(loc="lower right")
    plt.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, "roc_curve.png"))
    plt.close()
