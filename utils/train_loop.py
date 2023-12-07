from tqdm import tqdm, trange
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
import torch
from .plotting import save_metrics


def train_loop(model, train_loader, test_loader, device, num_layers, num_heads, hidden_dim, mlp_dim, N_EPOCHS, LR):
    train_loss_metric = []
    train_accuracy_metric = []
    test_loss_metric = []
    test_accuracy_metric = []

    optimizer = Adam(model.parameters(), lr=LR)
    criterion = CrossEntropyLoss()

    for epoch in trange(N_EPOCHS, desc="Training"):
        correct, total = 0, 0
        train_loss = 0.0
        # Train
        for batch in tqdm(
            train_loader, desc=f"Epoch {epoch + 1} in training", leave=False, miniters=1
        ):
            x, y = batch
            x, y = x.to(device), y.to(device)
            y_hat = model(x)
            loss = criterion(y_hat, y)

            train_loss += loss.detach().cpu().item() / len(train_loader)
            # print(f"\n\nTrain Loss: {train_loss}")

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            correct += torch.sum(torch.argmax(y_hat, dim=1)
                                 == y).detach().cpu().item()
            total += len(x)
            # print(f"[TRAIN] We have {correct} corrects and {total} total")

        print(
            f"Epoch {epoch + 1}/{N_EPOCHS} \nTrain loss: {train_loss:.2f}, Train accuracy: {correct / total * 100:.2f}%"
        )

        train_loss_metric.append(train_loss)
        train_accuracy_metric.append(correct / total * 100)

        # Test
        with torch.no_grad():
            correct_test, total_test = 0, 0
            test_loss = 0.0
            for batch in tqdm(test_loader, desc="Testing"):
                x, y = batch
                x, y = x.to(device), y.to(device)
                y_hat = model(x)
                loss = criterion(y_hat, y)

                test_loss += loss.detach().cpu().item() / len(test_loader)
                correct_test += (
                    torch.sum(torch.argmax(y_hat, dim=1)
                              == y).detach().cpu().item()
                )
                total_test += len(x)
            print(f"Test loss: {test_loss:.2f}")
            print(f"Test accuracy: {correct_test / total_test * 100:.2f}%")
            # print(f"[TEST] We have {correct_test} corrects and {total_test} total")
        test_loss_metric.append(test_loss)
        test_accuracy_metric.append(correct_test / total_test * 100)

    save_metrics(
        num_layers,
        num_heads,
        hidden_dim,
        mlp_dim,
        N_EPOCHS,
        LR,
        train_accuracy_metric,
        test_accuracy_metric,
        train_loss_metric,
        test_loss_metric
    )

    return (train_loss_metric, train_accuracy_metric, test_loss_metric, test_accuracy_metric)
