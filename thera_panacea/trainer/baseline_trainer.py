from pathlib import Path
from sklearn.metrics import confusion_matrix
from tqdm import tqdm
import torch
from torch.optim import Optimizer
from torch.nn import CrossEntropyLoss

from thera_panacea.model.baseline_model import BaselineModel


class BaselineTrainer:
    def __init__(
        self,
        model: BaselineModel,
        optimizer: Optimizer,
        loss_function: CrossEntropyLoss,
        model_path: Path = None
    ):
        self.model = model
        self.optimizer = optimizer
        self.loss_function = loss_function
        self.model_path = model_path
        self.best_hter = 0.5
        self.hter_not_improving_count = 0

    def train(self, train_dl, test_dl, n_epochs):
        epochs = range(1, n_epochs + 1)
        for epoch in epochs:
            print(f"Epoch {epoch} / {n_epochs}")

            self.model.train()
            for imgs, labels in tqdm(train_dl):
                self.fit(imgs, labels)

            print("Validation")
            self.model.eval()
            with torch.no_grad():
                hters = []
                for imgs, labels in tqdm(test_dl):
                    hter = self.eval(imgs, labels)
                    hters.append(hter)

            mean_hter = sum(hters)/len(hters)
            if mean_hter < self.best_hter:
                self.best_hter = mean_hter
                self.hter_not_improving_count = 0
                print(f"hter has improved : {mean_hter}")
                if self.model_path:
                    torch.save(self.model.state_dict(), self.model_path)
            else:
                self.hter_not_improving_count += 1

            if self.hter_not_improving_count == 5:
                break

    def fit(self, imgs, labels):
        self.optimizer.zero_grad()

        logits = self.model(imgs)
        loss: torch.Tensor = self.loss_function(logits, labels,)
        loss.backward()
        self.optimizer.step()

    def eval(self, imgs, labels):
        logits = self.model(imgs)
        preds = torch.argmax(logits, dim=-1).cpu()
        _, fp, fn, _ = confusion_matrix(
            labels.cpu(), preds,
            normalize="true").ravel()
        return (fp + fn) / 2
