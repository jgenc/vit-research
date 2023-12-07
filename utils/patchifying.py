from matplotlib import pyplot as plt


def show_patches(image):
    for i in range(0, 28, 4):
        xy1, xy2 = [i, 28], [i, i]
        plt.axline(xy1, xy2, c="white")
        xy1, xy2 = [28, i], [i, i]
        plt.axline(xy1, xy2, c="white")
    plt.axis("off")
    plt.imshow(image, cmap="gray", extent=(0, 28, 28, 0))
    plt.show()
