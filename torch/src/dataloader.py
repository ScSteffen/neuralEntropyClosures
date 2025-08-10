import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np


class MK11Dataset(Dataset):
    def __init__(self, csv_path: str, alpha_threshold: float = 10.0):
        """
        Loads data from CSV and preprocesses to tensors.
        Expects columns: u_0,...,u_5, alpha_0,...,alpha_5, h
        Filters out samples where alpha infinity norm exceeds alpha_threshold.
        """
        # Read CSV, skip timestamp column (first column)
        df = pd.read_csv(csv_path, skipinitialspace=True)

        # Extract columns
        u_cols = [f"u_{i}" for i in range(6)]
        alpha_cols = [f"alpha_{i}" for i in range(6)]

        # Convert to numpy arrays (exclude timestamp column)
        self.u = df[u_cols].values.astype(np.float32)[:, 1:]
        self.alpha = df[alpha_cols].values.astype(np.float32)[:, 1:]
        self.h = df["h"].values.astype(np.float32).reshape(-1, 1)  # shape (N,1)

        # Filter based on infinity norm of alpha
        alpha_linf_per_sample = np.max(np.abs(self.alpha), axis=1)
        valid_indices = alpha_linf_per_sample <= alpha_threshold

        # Keep only valid samples
        self.u = self.u[valid_indices]
        self.alpha = self.alpha[valid_indices]
        self.h = self.h[valid_indices]

        # Compute max norms for u and alpha on filtered data
        u_linf = np.max(np.abs(self.u))
        u_l1 = np.max(np.sum(np.abs(self.u), axis=1))
        u_l2 = np.max(np.linalg.norm(self.u, axis=1))

        alpha_linf = np.max(np.abs(self.alpha))
        alpha_l1 = np.max(np.sum(np.abs(self.alpha), axis=1))
        alpha_l2 = np.max(np.linalg.norm(self.alpha, axis=1))

        # Compute min, max for h
        h_min = np.min(self.h)
        h_max = np.max(self.h)
        self.h -= h_min  # shift h

        # Print statistics
        h_min = np.min(self.h)

        print(f"Dataset statistics (filtered with alpha_threshold={alpha_threshold}):")
        print(f"u max norms - L∞: {u_linf:.6f}, L1: {u_l1:.6f}, L2: {u_l2:.6f}")
        print(
            f"alpha max norms - L∞: {alpha_linf:.6f}, L1: {alpha_l1:.6f}, L2: {alpha_l2:.6f}"
        )
        print(f"h min value: {h_min:.6f}, max value: {h_max:.6f}")

    def __len__(self):
        return len(self.h)

    def __getitem__(self, idx):
        u_tensor = torch.from_numpy(self.u[idx])
        alpha_tensor = torch.from_numpy(self.alpha[idx])
        h_tensor = torch.from_numpy(self.h[idx])

        return u_tensor, alpha_tensor, h_tensor


def get_dataloader(csv_path: str, batch_size: int = 64, shuffle: bool = True):
    dataset = MK11Dataset(csv_path)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
