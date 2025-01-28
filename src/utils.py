# utils.py
import matplotlib.pyplot as plt

def plot_feature_importance(features, importances, title="Feature Importance"):
    plt.figure(figsize=(10, 6))
    plt.barh(features, importances)
    plt.xlabel("Importance")
    plt.ylabel("Features")
    plt.title(title)
    plt.show()
