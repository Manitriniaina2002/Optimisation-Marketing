import matplotlib.pyplot as plt


def bar_plot(series, title: str, path: str):
    ax = series.plot(kind='bar')
    ax.set_title(title)
    fig = ax.get_figure()
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
