#Task 2.2.2 and Task 2.2.3:


import numpy as np
from collections import deque

class EarlyStopping:

    def __init__(self, patience: int = 3, min_delta: float = 0.15, min_epochs: int = 20):
        self.patience = patience
        self.min_delta = min_delta
        self.min_epochs = min_epochs
        self.loss_history = deque(maxlen=patience)
        self.best_loss = float('inf')
        self.best_epoch = 0

    def __call__(self, trainer):

        epoch = trainer.epoch
        loss = trainer.loss
        self.loss_history.append(loss)

        if loss < self.best_loss:
            self.best_loss = loss
            self.best_epoch = epoch
            print(f"Эпоха {epoch}: новый минимум loss = {loss:.4f}")
        else:
            print(f"Эпоха {epoch}: loss = {loss:.4f} (лучший: {self.best_loss:.4f})")

        self.check_early_stopping(trainer)

    def check_early_stopping(self, trainer):
        epoch = trainer.epoch
        if epoch < self.min_epochs:
            return
        if len(self.loss_history) < self.patience:
            return
        recent_losses = list(self.loss_history)
        avg_recent_loss = np.mean(recent_losses)
        threshold = self.best_loss * (1 + self.min_delta)

        if avg_recent_loss > threshold:
            print(f"\nОстановка на эпохе {epoch}!")
            print(f"Средний loss за последние {self.patience} эпох: {avg_recent_loss:.4f}")
            print(f"Лучший loss: {self.best_loss:.4f} (порог: {threshold:.4f})")
            trainer.stop = True
        else:
            print(f"Обучение продолжается (avg loss: {avg_recent_loss:.4f} < порог: {threshold:.4f})")


def simple_callback(trainer):
    epoch = trainer.epoch
    loss = trainer.loss
    print(f"Эпоха: {epoch}, Loss: {loss}")


def create_early_stopping_callback(patience: int = 3, min_delta: float = 0.15, min_epochs: int = 20):
    return EarlyStopping(patience, min_delta, min_epochs)


#просто пример работы
if __name__ == "__main__":

    class MockTrainer:
        def __init__(self):
            self.epoch = 0
            self.loss = 0
            self.stop = False


    print("\n1")
    trainer = MockTrainer()
    early_stopping = EarlyStopping(patience=3, min_delta=0.15, min_epochs=5)

    losses = [2.0, 1.8, 1.5, 1.3, 1.1, 1.0, 0.9, 0.8, 0.75, 0.7]

    for epoch, loss in enumerate(losses):
        trainer.epoch = epoch
        trainer.loss = loss
        early_stopping(trainer)

        if trainer.stop:
            print(f"Остановлено на эпохе {epoch}")
            break

    if not trainer.stop:
        print("Обучение завершилось без ранней остановки")


    print("\n2")
    trainer = MockTrainer()
    early_stopping = EarlyStopping(patience=3, min_delta=0.15, min_epochs=5)

    losses = [2.0, 1.8, 1.5, 1.3, 1.2, 1.4, 1.6, 1.5, 1.7, 1.8]

    for epoch, loss in enumerate(losses):
        trainer.epoch = epoch
        trainer.loss = loss
        early_stopping(trainer)

        if trainer.stop:
            print(f"Остановлено на эпохе {epoch}")
            break

    print("\n3")
    trainer.epoch = 10
    trainer.loss = 1.234
    simple_callback(trainer)
