from __future__ import annotations

from typing import Callable, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        lr: float = 1e-3,
        ablation: str = "full",
    ):
        self.model = model.to(device)
        self.device = device
        self.ablation = ablation
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        self.criterion = nn.MSELoss()

    def _forward(self, x_endo, x_exo):
        if hasattr(self.model, "forward") and "ablation" in self.model.forward.__code__.co_varnames:
            return self.model(x_endo, x_exo, ablation=self.ablation)
        return self.model(x_endo, x_exo)

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        epochs: int = 15,
        patience: int = 5,
    ) -> list[float]:
        best_val = float("inf")
        stale = 0
        history = []

        for epoch in range(epochs):
            self.model.train()
            train_loss = 0.0
            for x_endo, x_exo, y in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False):
                x_endo, x_exo, y = x_endo.to(self.device), x_exo.to(self.device), y.to(self.device)
                self.optimizer.zero_grad()
                pred = self._forward(x_endo, x_exo)
                loss = self.criterion(pred, y)
                loss.backward()
                self.optimizer.step()
                train_loss += loss.item()
            train_loss /= max(len(train_loader), 1)

            val_loss = train_loss
            if val_loader is not None:
                self.model.eval()
                vloss = 0.0
                with torch.no_grad():
                    for x_endo, x_exo, y in val_loader:
                        x_endo, x_exo, y = x_endo.to(self.device), x_exo.to(self.device), y.to(self.device)
                        pred = self._forward(x_endo, x_exo)
                        vloss += self.criterion(pred, y).item()
                val_loss = vloss / max(len(val_loader), 1)

            history.append(val_loss)
            if val_loss < best_val:
                best_val = val_loss
                stale = 0
                self.best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
            else:
                stale += 1
                if stale >= patience:
                    break

        if hasattr(self, "best_state"):
            self.model.load_state_dict(self.best_state)
        return history
