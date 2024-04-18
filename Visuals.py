from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
import pandas as pd
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt


def graphic(input: pd.DataFrame, title: str, x= 10, y= 6):
    """
    Simple bar charts ploting
    """
    sns.set_palette(palette='RdPu')

    input = input.sort_values(ascending=True)

    size = (x, y)

    plt.figure(figsize=size)

    # Plotting the bar chart with seaborn's color palette
    input.plot(kind='barh', ylabel='')

    # Adding values on bars
    for i, value in enumerate(input):
        plt.text(value, i, str(value), ha='left', va='center')

    # Normalization
    norm = mcolors.Normalize(vmin=input.min(), vmax=input.max())

    # Plotting with the custom colormap
    plt.barh(input.index, input, 
            color=sns.color_palette('PuRd', n_colors=len(input)))

    plt.title(title)
    # Customize plot
    plt.gca().xaxis.set_ticks([])
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['left'].set_visible(False)
    plt.gca().spines['bottom'].set_visible(False)
    plt.grid(False)

    plt.show()