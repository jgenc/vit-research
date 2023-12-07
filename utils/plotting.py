import numpy as np
import matplotlib.pyplot as plt
import time
import os


def plot_results(
    num_layers,
    num_heads,
    hidden_dim,
    N_EPOCHS,
    LR,
    test_accuracy_epochs,
    train_accuracy_epochs,
    test_loss_epochs,
    train_loss_epochs,
):
    """
    Function that plots results from training.
    """

    epoch = np.arange(1, N_EPOCHS + 1)

    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    fig.suptitle(
        f"ViT with {num_layers} num_layers, {hidden_dim} hidden_dim, {num_heads} n_heads, {N_EPOCHS} epochs and LR {LR}"
    )

    axs[0].plot(
        epoch, np.array(test_accuracy_epochs), label="Test", c="red", marker="o"
    )
    axs[0].set_xlabel("Epoch")
    axs[0].set_ylabel("Accuracy")
    # Annotate last value on both lines
    axs[0].plot(
        epoch, np.array(train_accuracy_epochs), label="Train", c="blue", marker="o"
    )
    axs[0].legend()

    axs[1].plot(epoch, np.array(test_loss_epochs),
                label="Test", c="red", marker="o")
    axs[1].set_xlabel("Epoch")
    axs[1].set_ylabel("Loss")
    axs[1].plot(epoch, np.array(train_loss_epochs),
                label="Train", c="blue", marker="o")
    axs[1].legend()
    fig.show()


def show_metrics(filename):
    """
    Plots the metrics from a file
    """

    (
        train_accuracy,
        test_accuracy,
        train_loss,
        test_loss,
        parameters,
    ) = np.load(filename, allow_pickle=True).values()
    # for i in np.load(filename, allow_pickle=True).values():
    #     print(i)
    parameters = parameters.item()
    num_layers = parameters["num_layers"]
    num_heads = parameters["num_heads"]
    hidden_dim = parameters["hidden_dim"]
    mlp_dim = parameters["mlp_dim"]
    N_EPOCHS = parameters["N_EPOCHS"]
    LR = parameters["LR"]

    plot_results(
        num_layers,
        num_heads,
        hidden_dim,
        N_EPOCHS,
        LR,
        test_accuracy,
        train_accuracy,
        test_loss,
        train_loss,
    )
    print(parameters)
    print(
        "Train accuracy: ",
        train_accuracy[-1],
        "Train loss: ",
        train_loss[-1],
    )
    print(
        "Test accuracy: ",
        test_accuracy[-1],
        "Test loss: ",
        test_loss[-1],
    )


def save_metrics(
    num_layers,
    num_heads,
    hidden_dim,
    mlp_dim,
    N_EPOCHS,
    LR,
    train_accuracy,
    test_accuracy,
    train_loss,
    test_loss,
):
    """
    Saving metrics to an npz file.
    """
    time_now = time.localtime()
    date = time.strftime("%y-%m-%d_%H-%M-%S", time_now)
    name = f"{date}_metrics.npz"
    parameters = {
        "num_layers": num_layers,
        "num_heads": num_heads,
        "hidden_dim": hidden_dim,
        "mlp_dim": mlp_dim,
        "N_EPOCHS": N_EPOCHS,
        "LR": LR,
    }
    np.savez(
        os.path.join("metrics", name),
        train_accuracy,
        test_accuracy,
        train_loss,
        test_loss,
        parameters,
    )
