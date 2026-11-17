import tensorflow as tf

from src.training.train_config import TrainingConfig


def build_optimizer(model, config: TrainingConfig):
    if not isinstance(config, TrainingConfig):
        raise TypeError("config must be a TrainingConfig instance.")

    if not model.trainable_variables:
        raise ValueError("The model has no trainable variables.")

    if config.optimizer == "adam":
        return tf.keras.optimizers.Adam(
            learning_rate=config.learning_rate
        )

    if config.optimizer == "adamw":
        return tf.keras.optimizers.AdamW(
            learning_rate=config.learning_rate,
            weight_decay=config.weight_decay
        )

    if config.optimizer == "sgd":
        return tf.keras.optimizers.SGD(
            learning_rate=config.learning_rate,
            momentum=0.9
        )

    raise ValueError(f"Unsupported optimizer: {config.optimizer}")


def build_scheduler(optimizer, config: TrainingConfig):
    if not isinstance(config, TrainingConfig):
        raise TypeError("config must be a TrainingConfig instance.")

    if config.scheduler == "none":
        return None

    if config.scheduler == "plateau":
        return tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=config.scheduler_gamma,
            patience=config.scheduler_patience,
            min_lr=config.min_learning_rate,
            mode="min"
        )

    def schedule(epoch, current_lr):
        if config.scheduler == "step":
            exponent = (epoch + 1) // config.scheduler_step_size
            new_lr = config.learning_rate * (
                config.scheduler_gamma ** exponent
            )
            return max(new_lr, config.min_learning_rate)

        if config.scheduler == "cosine":
            import math

            progress = min(epoch / max(config.epochs, 1), 1.0)
            cosine = 0.5 * (1.0 + math.cos(math.pi * progress))

            return (
                config.min_learning_rate
                + (config.learning_rate - config.min_learning_rate)
                * cosine
            )

        raise ValueError(
            f"Unsupported scheduler: {config.scheduler}"
        )

    return tf.keras.callbacks.LearningRateScheduler(
        schedule,
        verbose=0
    )


if __name__ == "__main__":
    print("Optimizer and Scheduler Tests")
    print("-" * 45)

    config = TrainingConfig()

    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(10,)),
        tf.keras.layers.Dense(5)
    ])

    optimizer = build_optimizer(model, config)
    print("Optimizer:", type(optimizer).__name__)

    scheduler = build_scheduler(optimizer, config)
    print("Scheduler:", type(scheduler).__name__)

    x = tf.random.normal([4, 10])
    y = tf.random.normal([4, 5])

    with tf.GradientTape() as tape:
        predictions = model(x)
        loss = tf.reduce_mean(tf.square(predictions - y))

    gradients = tape.gradient(
        loss,
        model.trainable_variables
    )

    optimizer.apply_gradients(
        zip(gradients, model.trainable_variables)
    )

    print("Optimizer update: PASS")

    if scheduler is not None:
        assert isinstance(scheduler, tf.keras.callbacks.Callback)
        print("Scheduler callback: PASS")

    print("-" * 45)
    print("All Optimizer tests passed!")