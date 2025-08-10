import torch
from torch import nn, optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from src.mk11 import MK11
from src.dataloader import MK11Dataset, get_dataloader
import wandb

wandb_flag = True


def train_mk11(
    model: nn.Module,
    train_loader,
    epochs: int = 2000,
    lr: float = 1e-3,
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
):
    if wandb_flag:
        wandb.init(project="MK11_training", config={"epochs": epochs, "lr": lr})
        config = wandb.config

    model.to(device)
    model.train()

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    for n, p in model.named_parameters():
        print(p.shape, n, p.requires_grad)

    scheduler = CosineAnnealingLR(
        optimizer, T_max=epochs * len(train_loader), eta_min=1e-6
    )
    criterion = nn.MSELoss()

    for epoch in range(1, epochs + 1):
        running_loss = 0.0
        loop = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}")

        for batch_idx, (u, alpha, h) in enumerate(loop):
            u = u.to(device)
            alpha = alpha.to(device)
            h = h.to(device)

            optimizer.zero_grad()

            # Forward pass
            h_pred, alpha_pred = model(u)
            # print(h, h_pred)

            # Compute losses
            loss_h = criterion(h_pred, h)
            loss_alpha = criterion(alpha_pred, alpha)

            loss = loss_h + loss_alpha

            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            avg_loss = running_loss / (batch_idx + 1)
            loop.set_postfix(loss=avg_loss, lr=optimizer.param_groups[0]["lr"])

            # Optional: log batch loss to wandb
            if wandb_flag:
                wandb.log(
                    {
                        "batch_loss": loss.item(),
                        "loss_h": loss_h.item(),
                        "loss_alpha": loss_alpha.item(),
                        "learning_rate": optimizer.param_groups[0]["lr"],
                        "epoch": epoch,
                        "batch": batch_idx,
                    }
                )

            scheduler.step()

        # Log epoch summary loss and lr
        if wandb_flag:
            wandb.log(
                {
                    "epoch_loss": avg_loss,
                    "learning_rate": optimizer.param_groups[0]["lr"],
                    "epoch": epoch,
                }
            )
        if epoch % 10 == 0:
            save_model(
                model,
                f"models/mk11_trained_model_width_{model.core_model.width}_depth_{model.core_model.depth}_epoch_{epoch}_loss_{avg_loss}.pth",
            )
            # wandb.save(f"mk11_epoch_{epoch}_loss_{avg_loss}.pth")

    print("Training completed.")
    wandb.finish()


def load_model(
    model: nn.Module, path: str, device: torch.device = torch.device("cpu")
) -> nn.Module:
    """
    Loads the model's state_dict from the given path and moves model to device.
    """
    state_dict = torch.load(path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()  # set to eval mode by default after loading
    print(f"Model loaded from {path} and moved to {device}")
    return model


def save_model(model: nn.Module, path: str):
    """
    Saves the model's state_dict to the given path.
    """
    torch.save(model.state_dict(), path)
    print(f"Model saved to {path}")


def main():
    # Hyperparameters
    epochs = 2000
    batch_size = 512
    learning_rate = 1e-3
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load real dataset from CSV
    data_path = "../data/2D/SphericalHarmonics_M2_2D_normal_gaussian_gamma1.csv"
    dataset = MK11Dataset(data_path)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Initialize model - adjust args as needed for your MK11 class
    model = MK11(polynomial_degree=2, width=512, depth=4, gamma=1)
    # model = SimpleResNet().cuda()
    # model = load_model(model, "mk11_trained_model.pth", device=device)
    # Train the model
    train_mk11(model, train_loader, epochs=epochs, lr=learning_rate, device=device)

    # Save the trained model
    save_model(
        model,
        f"models/mk11_trained_model_width_{model.core_model.width}_depth_{model.core_model.depth}.pth",
    )


if __name__ == "__main__":
    main()
