
import torch
import torch.optim as optim

from torch.optim.lr_scheduler import (
    StepLR,
    CosineAnnealingLR,
    ReduceLROnPlateau,
)

from src.training.train_config import TrainingConfig


def build_optimizer(model, config: TrainingConfig):
    """
    Create an optimizer using trainable model parameters only.

    Supported optimizers:
        - Adam
        - AdamW
        - SGD
    """

    if not isinstance(config, TrainingConfig):
        raise TypeError("config must be a TrainingConfig instance.")

    trainable_parameters = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad
    ]

    if not trainable_parameters:
        raise ValueError(
            "The model has no trainable parameters."
        )

    if config.optimizer == "adam":
        optimizer = optim.Adam(
            trainable_parameters,
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )

    elif config.optimizer == "adamw":
        optimizer = optim.AdamW(
            trainable_parameters,
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )

    elif config.optimizer == "sgd":
        optimizer = optim.SGD(
            trainable_parameters,
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
            momentum=0.9,
        )

    else:
        raise ValueError(
            f"Unsupported optimizer: {config.optimizer}"
        )

    return optimizer


def build_scheduler(optimizer, config: TrainingConfig):
    """
    Create a learning rate scheduler.

    Supported schedulers:
        - None
        - StepLR
        - CosineAnnealingLR
        - ReduceLROnPlateau
    """

    if not isinstance(config, TrainingConfig):
        raise TypeError("config must be a TrainingConfig instance.")

    if config.scheduler == "none":
        return None

    elif config.scheduler == "step":
        scheduler = StepLR(
            optimizer,
            step_size=config.scheduler_step_size,
            gamma=config.scheduler_gamma,
        )

    elif config.scheduler == "cosine":
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=config.epochs,
            eta_min=config.min_learning_rate,
        )

    elif config.scheduler == "plateau":
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=config.scheduler_gamma,
            patience=config.scheduler_patience,
            min_lr=config.min_learning_rate,
        )

    else:
        raise ValueError(
            f"Unsupported scheduler: {config.scheduler}"
        )

    return scheduler


if __name__ == "__main__":
    # Simple integration smoke test
    model = torch.nn.Linear(10, 5)

    config = TrainingConfig()

    optimizer = build_optimizer(model, config)
    scheduler = build_scheduler(optimizer, config)

    print("Optimizer:", type(optimizer).__name__)
    print("Scheduler:", type(scheduler).__name__)

    print(
        "Trainable parameters:",
        sum(
            p.numel()
            for p in model.parameters()
            if p.requires_grad
        ),
    )

    print("Initial learning rate:", optimizer.param_groups[0]["lr"])

    # Verify optimizer update
    dummy_input = torch.randn(4, 10)
    dummy_target = torch.randn(4, 5)

    criterion = torch.nn.MSELoss()

    optimizer.zero_grad()

    output = model(dummy_input)
    loss = criterion(output, dummy_target)

    loss.backward()
    optimizer.step()

    print("Optimizer update: PASS")

    if scheduler is not None:
        if isinstance(scheduler, ReduceLROnPlateau):
            scheduler.step(loss.item())
        else:
            scheduler.step()

        print("Scheduler step: PASS")
