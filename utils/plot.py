import matplotlib.pyplot as plt


def white_2d_plot():
    ax = plt.subplot(111)
    ax.tick_params(axis='x', colors='white')
    ax.tick_params(axis='y', colors='white')
    return ax
