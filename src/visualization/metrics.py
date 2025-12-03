"""Metric plotting helpers."""

import matplotlib.pyplot as plt


def plot_curve(
    values: list[float],
    title: str,
    *,
    save_path: str | None = None,
    show: bool = True,
) -> None:
    fig, ax = plt.subplots()
    ax.plot(values)
    ax.set_title(title)
    ax.set_xlabel("Step")
    ax.set_ylabel("Value")
    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path)
        plt.close(fig)
        return

    if show:
        plt.show()
    else:
        plt.close(fig)
