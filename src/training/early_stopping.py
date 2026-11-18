
import math


class EarlyStopping:
    """
    Stop training when validation loss stops improving.
    """

    def __init__(
        self,
        patience=7,
        min_delta=0.001,
    ):
        if (
            not isinstance(patience, int)
            or isinstance(patience, bool)
            or patience < 0
        ):
            raise ValueError(
                "patience must be a non-negative integer."
            )

        if (
            isinstance(min_delta, bool)
            or not isinstance(min_delta, (int, float))
        ):
            raise TypeError(
                "min_delta must be numeric."
            )

        if (
            not math.isfinite(float(min_delta))
            or min_delta < 0
        ):
            raise ValueError(
                "min_delta must be finite and non-negative."
            )

        self.patience = patience
        self.min_delta = float(min_delta)

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

        if isinstance(validation_loss, bool) or not isinstance(
            validation_loss, (int, float)
        ):
            raise TypeError(
                "validation_loss must be numeric."
            )

        validation_loss = float(validation_loss)

        if not math.isfinite(validation_loss):
            raise ValueError(
                "validation_loss must be finite."
            )

        if self.early_stop:
            return True

        improved = (
            validation_loss < self.best_loss - self.min_delta
        )

        if improved:
            self.best_loss = validation_loss
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

    def state_dict(self):
        """Return the current Early Stopping state."""

        return {
            "best_loss": self.best_loss,
            "counter": self.counter,
            "early_stop": self.early_stop,
        }

    def load_state_dict(self, state):
        """Restore a previously saved Early Stopping state."""

        if not isinstance(state, dict):
            raise TypeError(
                "state must be a dictionary."
            )

        required = {
            "best_loss",
            "counter",
            "early_stop",
        }

        missing = required - state.keys()

        if missing:
            raise KeyError(
                f"Missing Early Stopping state keys: {missing}"
            )

        best_loss = state["best_loss"]
        counter = state["counter"]
        early_stop = state["early_stop"]

        if (
            isinstance(best_loss, bool)
            or not isinstance(best_loss, (int, float))
            or math.isnan(float(best_loss))
            or float(best_loss) == float("-inf")
        ):
            raise ValueError(
                "Invalid best_loss in Early Stopping state."
            )

        if (
            not isinstance(counter, int)
            or isinstance(counter, bool)
            or counter < 0
        ):
            raise ValueError(
                "counter must be a non-negative integer."
            )

        if not isinstance(early_stop, bool):
            raise TypeError(
                "early_stop must be boolean."
            )

        self.best_loss = float(best_loss)
        self.counter = counter
        self.early_stop = early_stop

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

    saved_state = early_stopping.state_dict()

    restored = EarlyStopping(
        patience=3,
        min_delta=0.001,
    )

    restored.load_state_dict(saved_state)

    assert restored.best_loss == early_stopping.best_loss
    assert restored.counter == early_stopping.counter
    assert restored.early_stop == early_stopping.early_stop

    print("\nEarlyStopping test: PASS")
