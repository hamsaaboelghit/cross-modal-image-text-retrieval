
class EarlyStopping:
    """
    Stop training when validation loss stops improving.

    Parameters:
        patience:
            Number of consecutive epochs without improvement.

        min_delta:
            Minimum improvement required to reset the counter.
    """

    def __init__(
        self,
        patience=7,
        min_delta=0.001,
    ):
        if patience < 0:
            raise ValueError(
                "patience cannot be negative."
            )

        if min_delta < 0:
            raise ValueError(
                "min_delta cannot be negative."
            )

        self.patience = patience
        self.min_delta = min_delta

        self.best_loss = float("inf")
        self.counter = 0
        self.early_stop = False

    def __call__(self, validation_loss):
        """
        Update Early Stopping state.

        Returns:
            True if training should stop.
            False otherwise.
        """

        if not isinstance(validation_loss, (int, float)):
            raise TypeError(
                "validation_loss must be numeric."
            )

        if not (
            float("-inf") < validation_loss < float("inf")
        ):
            raise ValueError(
                "validation_loss must be finite."
            )

        if validation_loss < self.best_loss - self.min_delta:

            self.best_loss = float(validation_loss)
            self.counter = 0

            print(
                f"Validation loss improved: "
                f"{self.best_loss:.6f}"
            )

        else:
            self.counter += 1

            print(
                f"No improvement: "
                f"{self.counter}/{self.patience}"
            )

            if self.counter >= self.patience:
                self.early_stop = True

                print("Early stopping triggered!")

        return self.early_stop

    def reset(self):
        """Reset Early Stopping state."""

        self.best_loss = float("inf")
        self.counter = 0
        self.early_stop = False


if __name__ == "__main__":

    early_stopping = EarlyStopping(
        patience=3,
        min_delta=0.001,
    )

    # Synthetic values for testing only.
    validation_losses = [
        0.90,
        0.85,
        0.86,
        0.87,
        0.88,
    ]

    print("\nEarly Stopping Test")
    print("-" * 35)

    for epoch, loss in enumerate(
        validation_losses,
        start=1,
    ):
        print(f"\nEpoch {epoch}")

        should_stop = early_stopping(loss)

        if should_stop:
            break

    assert early_stopping.early_stop
    assert early_stopping.best_loss == 0.85

    print("\nEarlyStopping test: PASS")
