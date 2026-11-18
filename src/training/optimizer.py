
import math

import tensorflow as tf

from src.training.train_config import TrainingConfig


def _validate_config(config):
    """Validate the training configuration used by the optimizer."""
    if not isinstance(config, TrainingConfig):
        raise TypeError("config must be a TrainingConfig instance.")

    if config.learning_rate <= 0:
        raise ValueError("learning_rate must be greater than zero.")

    if config.weight_decay < 0:
        raise ValueError("weight_decay cannot be negative.")

    if config.min_learning_rate < 0:
        raise ValueError("min_learning_rate cannot be negative.")

    if config.min_learning_rate > config.learning_rate:
        raise ValueError(
            "min_learning_rate cannot exceed learning_rate."
        )

    if config.scheduler_gamma <= 0 or config.scheduler_gamma > 1:
        raise ValueError(
            "scheduler_gamma must be greater than 0 and at most 1."
        )

    if config.scheduler_step_size <= 0:
        raise ValueError(
            "scheduler_step_size must be greater than zero."
        )

    if config.epochs <= 0:
        raise ValueError("epochs must be greater than zero.")


def build_optimizer(model, config: TrainingConfig):
    """
    Build a TensorFlow optimizer.

    Supported optimizers:
        - Adam
        - AdamW
        - SGD

    The optimizer updates the trainable variables supplied by the model
    during the training loop.
    """
    _validate_config(config)

    if model is None:
        raise ValueError("model cannot be None.")

    if not hasattr(model, "trainable_variables"):
        raise TypeError(
            "model must expose trainable_variables."
        )

    if not model.trainable_variables:
        raise ValueError(
            "The model has no trainable variables. "
            "Build the model before creating the optimizer."
        )

    if config.optimizer == "adam":
        optimizer = tf.keras.optimizers.Adam(
            learning_rate=config.learning_rate
        )

    elif config.optimizer == "adamw":
        optimizer = tf.keras.optimizers.AdamW(
            learning_rate=config.learning_rate,
            weight_decay=config.weight_decay
        )

    elif config.optimizer == "sgd":
        optimizer = tf.keras.optimizers.SGD(
            learning_rate=config.learning_rate,
            momentum=0.9
        )

    else:
        raise ValueError(
            f"Unsupported optimizer: {config.optimizer}. "
            "Choose adam, adamw, or sgd."
        )

    return optimizer


def build_scheduler(optimizer, config: TrainingConfig):
    """
    Build a Keras learning-rate scheduler callback.

    Supported schedulers:
        - none
        - step
        - cosine
        - plateau

    The returned callback is controlled by Trainer.fit().
    """
    _validate_config(config)

    if optimizer is None:
        raise ValueError("optimizer cannot be None.")

    if config.scheduler == "none":
        return None

    if config.scheduler == "plateau":

        if config.scheduler_gamma >= 1.0:
            raise ValueError(
                "For the plateau scheduler, scheduler_gamma "
                "must be less than 1."
            )

        return tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=config.scheduler_gamma,
            patience=config.scheduler_patience,
            min_lr=config.min_learning_rate,
            mode="min",
            verbose=1
        )

    if config.scheduler == "step":

        def step_schedule(epoch, current_lr):
            """
            Reduce the learning rate after every step_size epochs.
            """
            exponent = epoch // config.scheduler_step_size

            new_lr = (
                config.learning_rate
                * (config.scheduler_gamma ** exponent)
            )

            return max(
                new_lr,
                config.min_learning_rate
            )

        return tf.keras.callbacks.LearningRateScheduler(
            step_schedule,
            verbose=0
        )

    if config.scheduler == "cosine":

        def cosine_schedule(epoch, current_lr):
            """
            Cosine annealing from the initial learning rate
            to the minimum learning rate.

            Epoch 0:
                Initial learning rate.

            Last epoch:
                Minimum learning rate.
            """
            if config.epochs == 1:
                return config.min_learning_rate

            progress = min(
                max(epoch, 0) / (config.epochs - 1),
                1.0
            )

            cosine_value = 0.5 * (
                1.0 + math.cos(math.pi * progress)
            )

            new_lr = (
                config.min_learning_rate
                + (
                    config.learning_rate
                    - config.min_learning_rate
                ) * cosine_value
            )

            return max(
                new_lr,
                config.min_learning_rate
            )

        return tf.keras.callbacks.LearningRateScheduler(
            cosine_schedule,
            verbose=0
        )

    raise ValueError(
        f"Unsupported scheduler: {config.scheduler}. "
        "Choose none, step, cosine, or plateau."
    )


if __name__ == "__main__":

    print("=" * 55)
    print("VISIONMATCH AI - OPTIMIZER TEST")
    print("=" * 55)

    config = TrainingConfig()

    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(10,)),
        tf.keras.layers.Dense(5)
    ])

    optimizer = build_optimizer(model, config)

    print(
        "Optimizer:",
        type(optimizer).__name__
    )

    scheduler = build_scheduler(
        optimizer,
        config
    )

    print(
        "Scheduler:",
        type(scheduler).__name__
        if scheduler is not None
        else "None"
    )

    # Test a real optimizer update.
    x = tf.random.normal((4, 10))
    y = tf.random.normal((4, 5))

    with tf.GradientTape() as tape:

        predictions = model(x)

        loss = tf.reduce_mean(
            tf.square(predictions - y)
        )

    gradients = tape.gradient(
        loss,
        model.trainable_variables
    )

    optimizer.apply_gradients(
        zip(
            gradients,
            model.trainable_variables
        )
    )

    print("Optimizer update: PASS")

    # Test the scheduler callback.
    if scheduler is not None:

        if not isinstance(
            scheduler,
            tf.keras.callbacks.Callback
        ):
            raise AssertionError(
                "Scheduler must be a Keras callback."
            )

        print("Scheduler callback: PASS")

    print("=" * 55)
    print("Optimizer tests completed.")
