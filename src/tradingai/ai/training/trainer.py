"""Bucle de entrenamiento multi-tarea: direccion/entry_timeframe (clasificacion) + entry/tp/sl (regresion)."""

from __future__ import annotations

from pathlib import Path

import torch
from loguru import logger
from torch import nn
from torch.utils.data import DataLoader, Dataset

from tradingai.ai.data.multi_timeframe import TIMEFRAMES
from tradingai.ai.models.base import MultiTimeframeTradingModel


class MultiTaskLoss(nn.Module):
    """Combina cross-entropy (direccion, entry_timeframe) con MSE (entry/tp/sl).

    La regresion de entry/tp/sl y la clasificacion de entry_timeframe solo tienen
    sentido cuando hay una direccion operable (no neutral).
    """

    def __init__(
        self,
        direction_weight: float = 1.0,
        regression_weight: float = 0.5,
        entry_tf_weight: float = 0.3,
        direction_class_weights: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        self.direction_weight = direction_weight
        self.regression_weight = regression_weight
        self.entry_tf_weight = entry_tf_weight
        # Ponderado por frecuencia inversa: sin esto, un dataset con clases
        # desbalanceadas (p.ej. mas "short" que "long"/"neutral") tiende a colapsar
        # a predecir siempre la clase mayoritaria, sobre todo en pocas epochs.
        self.ce = nn.CrossEntropyLoss(weight=direction_class_weights)
        self.ce_entry_tf = nn.CrossEntropyLoss()
        self.mse = nn.MSELoss()

    def forward(self, outputs: dict, batch: dict) -> torch.Tensor:
        direction_loss = self.ce(outputs["direction_logits"], batch["direction"])

        tradeable = batch["direction"] != 0
        if tradeable.any():
            entry_loss = self.mse(outputs["entry_offset"][tradeable], batch["entry_offset"][tradeable])
            tp_loss = self.mse(outputs["tp_mult"][tradeable], batch["tp_mult"][tradeable])
            sl_loss = self.mse(outputs["sl_mult"][tradeable], batch["sl_mult"][tradeable])
            regression_loss = entry_loss + tp_loss + sl_loss
            entry_tf_loss = self.ce_entry_tf(
                outputs["entry_timeframe_logits"][tradeable], batch["entry_timeframe"][tradeable]
            )
        else:
            regression_loss = torch.tensor(0.0, device=direction_loss.device)
            entry_tf_loss = torch.tensor(0.0, device=direction_loss.device)

        return (
            self.direction_weight * direction_loss
            + self.regression_weight * regression_loss
            + self.entry_tf_weight * entry_tf_loss
        )


class Trainer:
    def __init__(
        self,
        model: MultiTimeframeTradingModel,
        config: dict,
        device: str | None = None,
        direction_class_weights: torch.Tensor | None = None,
    ) -> None:
        self.model = model
        self.config = config
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        model_cfg = config["model"]
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=model_cfg.get("learning_rate", 1e-4))
        if direction_class_weights is not None:
            direction_class_weights = direction_class_weights.to(self.device)
        self.criterion = MultiTaskLoss(direction_class_weights=direction_class_weights)
        self.epochs = model_cfg.get("epochs", 100)
        self.patience = model_cfg.get("early_stopping_patience", 10)

    def fit(self, train_dataset: Dataset, val_dataset: Dataset | None = None) -> None:
        batch_size = self.config["model"].get("batch_size", 64)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size) if val_dataset else None

        best_val_loss = float("inf")
        best_state_dict = None
        epochs_without_improvement = 0

        for epoch in range(1, self.epochs + 1):
            train_loss = self._run_epoch(train_loader, train=True)
            log_msg = f"Epoch {epoch}/{self.epochs} - train_loss={train_loss:.4f}"

            if val_loader:
                val_loss = self._run_epoch(val_loader, train=False)
                log_msg += f" - val_loss={val_loss:.4f}"

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    # Copia en CPU: no queremos mantener una segunda copia en GPU si
                    # hay CUDA disponible, y state_dict() ya devuelve tensores nuevos.
                    best_state_dict = {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}
                    epochs_without_improvement = 0
                else:
                    epochs_without_improvement += 1
                    if epochs_without_improvement >= self.patience:
                        logger.info(f"Early stopping en epoch {epoch} (sin mejora en {self.patience} epochs)")
                        break

            logger.info(log_msg)

        if best_state_dict is not None:
            self.model.load_state_dict(best_state_dict)
            logger.info(f"Restaurados los pesos de la mejor epoch (val_loss={best_val_loss:.4f})")

    def _run_epoch(self, loader: DataLoader, train: bool) -> float:
        self.model.train(mode=train)
        total_loss = 0.0
        n_examples = 0

        for batch in loader:
            batch = {k: v.to(self.device) for k, v in batch.items()}
            sequences = {tf: batch.pop(tf) for tf in TIMEFRAMES}
            batch_size = sequences["M15"].size(0)

            with torch.set_grad_enabled(train):
                outputs = self.model(sequences)
                loss = self.criterion(outputs, batch)

                if train:
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

            total_loss += loss.item() * batch_size
            n_examples += batch_size

        return total_loss / n_examples

    def save_checkpoint(self, path: str | Path, feature_columns: list[str]) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        # "secrets" (credenciales/URL del bridge) no hace falta para inferencia y no es
        # un tipo puro de datos (pydantic BaseSettings) — rompe la carga con
        # weights_only=True (default desde PyTorch 2.6) si se guarda en el checkpoint.
        config_without_secrets = {k: v for k, v in self.config.items() if k != "secrets"}
        torch.save(
            {
                "model_state": self.model.state_dict(),
                "config": config_without_secrets,
                "feature_columns": feature_columns,
            },
            path,
        )
        logger.info(f"Checkpoint guardado en {path}")
