import seaborn as sns
import matplotlib.pyplot as plt

def plot_correlation_heatmap(data):
    """Plot a correlation heatmap."""
    plt.figure(figsize=(10, 8))
    sns.heatmap(data.corr(), annot=True, cmap="YlGnBu")
    plt.title("Correlation Heatmap")
    plt.show()
