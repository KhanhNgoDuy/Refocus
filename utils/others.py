from matplotlib import pyplot as plt
import yaml

def load_config_file(config_file):
    with open(config_file, "r") as f:
        configs = yaml.safe_load(f)
    return configs


def show_plot(obj, title: str, cmap=None):
    plt.imshow(obj, cmap=cmap)
    plt.title(title)
    plt.show()
    plt.close()