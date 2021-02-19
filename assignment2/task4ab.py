import utils
import matplotlib.pyplot as plt
from task2a import pre_process_images, one_hot_encode, SoftmaxModel
from task2 import SoftmaxTrainer


if __name__ == "__main__":
    # hyperparameters DO NOT CHANGE IF NOT SPECIFIED IN ASSIGNMENT TEXT
    num_epochs = 50
    learning_rate = .02
    batch_size = 32
    neurons_per_layer = [32, 10]
    momentum_gamma = .9  # Task 3 hyperparameter
    shuffle_data = True

    use_improved_sigmoid = True
    use_improved_weight_init = True
    use_momentum = True

    # Load dataset
    X_train, Y_train, X_val, Y_val = utils.load_full_mnist()
    X_train = pre_process_images(X_train)
    X_val = pre_process_images(X_val)
    Y_train = one_hot_encode(Y_train, 10)
    Y_val = one_hot_encode(Y_val, 10)

    # 32 neurons per layer
    model_a = SoftmaxModel(
        neurons_per_layer,
        use_improved_sigmoid,
        use_improved_weight_init)
    trainer_a = SoftmaxTrainer(
        momentum_gamma, use_momentum,
        model_a, learning_rate, batch_size, shuffle_data,
        X_train, Y_train, X_val, Y_val,
    )
    train_history_a, val_history_a = trainer_a.train(num_epochs)

    # 128 neurons per layer
    neurons_per_layer = [128, 10]
    model_b = SoftmaxModel(
        neurons_per_layer,
        use_improved_sigmoid,
        use_improved_weight_init)
    trainer_b = SoftmaxTrainer(
        momentum_gamma, use_momentum,
        model_b, learning_rate, batch_size, shuffle_data,
        X_train, Y_train, X_val, Y_val,
    )
    train_history_b, val_history_b = trainer_b.train(num_epochs)


    plt.figure(figsize=(25,15))
    plt.subplot(1, 2, 1)
    plt.ylim([0, .8])
    utils.plot_loss(
        train_history_a["loss"], "Task 4 Model a - 32 epochs", npoints_to_average=10)
    utils.plot_loss(
        train_history_b["loss"], "Task 4 Model b - 128 epochs", npoints_to_average=10)
    plt.ylabel("Training loss")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.ylim([0.85, 1.0])
    utils.plot_loss(
        val_history_a["accuracy"], "Task 4 Model a - 32 epochs")
    utils.plot_loss(
        val_history_b["accuracy"], "Task 4 Model b - 128 epochs")
    plt.ylabel("Validation Accuracy")
    plt.legend()
    plt.savefig("task4_comparison.png")
    #plt.show()
