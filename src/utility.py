import matplotlib.pyplot as plt

def save_image(fig, filename):
    fig.tight_layout()
    plt.savefig(filename, dpi=300)