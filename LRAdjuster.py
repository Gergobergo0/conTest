import math

class LRAdjuster:
    def __init__(self, optimizer, init_lr=0.001, min_lr=0.0001, patience=10, factor=0.5, tolerance=0.01):
        """
        Improved LR Adjuster with decay stabilization and adaptive validation monitoring.

        :param optimizer: Optimizer instance to adjust learning rate.
        :param init_lr: Initial learning rate.
        :param min_lr: Minimum learning rate.
        :param patience: Number of epochs to wait before reducing LR if no improvement.
        :param factor: Factor for LR reduction when patience is exceeded.
        :param tolerance: Minimum validation accuracy improvement to reset patience.
        """
        self.optimizer = optimizer
        self.init_lr = init_lr
        self.min_lr = min_lr
        self.patience = patience
        self.factor = factor
        self.tolerance = tolerance

        self.best_val_loss = float('inf')
        self.num_bad_epochs = 0
        self.lr = init_lr

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.lr

    def step(self, val_loss):
        """
        Update learning rate based on validation loss performance.

        :param val_loss: Current validation loss.
        """
        if val_loss < self.best_val_loss - self.tolerance:
            # Improvement in validation loss
            self.best_val_loss = val_loss
            self.num_bad_epochs = 0
        else:
            # No improvement
            self.num_bad_epochs += 1

        if self.num_bad_epochs >= self.patience:
            # Reduce learning rate if no improvement for `patience` epochs
            new_lr = max(self.lr * self.factor, self.min_lr)
            if new_lr < self.lr:  # Only update if new LR is lower
                self.lr = new_lr
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = self.lr
                print(f"Reducing learning rate to {self.lr:.6f}")
            self.num_bad_epochs = 0

    def reset(self):
        """
        Reset the LRAdjuster to its initial state.
        """
        self.best_val_loss = float('inf')
        self.num_bad_epochs = 0
        self.lr = self.init_lr

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.lr

        print("LRAdjuster reset to initial state.")
