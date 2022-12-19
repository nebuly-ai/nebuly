from abc import ABC, abstractmethod

import torch
from nebullvm.operations.base import Operation
from nebullvm.operations.fetch_operations.local import FetchModelFromLocal
from torch.utils.data import DataLoader
from torchvision import datasets


from forward_forward.operations.data import VOCABULARY
from forward_forward.operations.fetch_operations import (
    FetchTrainingDataFromLocal,
)
from forward_forward.utils import (
    ProgressiveTrainingDataset,
    compute_perplexity,
)
from forward_forward.utils.labels import LabelsInjector
from forward_forward.utils.modules import FCNetFFProgressive


class BaseForwardForwardTrainer(Operation, ABC):
    def __init__(self):
        super().__init__()
        self.model = None
        self.train_data = None
        self.test_data = None

        self.fetch_model_op = FetchModelFromLocal()
        self.fetch_data_op = FetchTrainingDataFromLocal()

    def get_result(self):
        if self.state.get("model_is_trained"):
            return self.model

    def execute(
        self,
        model: FCNetFFProgressive,
        train_data: DataLoader,
        test_data: DataLoader,
        epochs: int,
        theta: float,
        **kwargs,
    ):
        if self.fetch_model_op.get_model() is None:
            self.fetch_model_op.execute(model)
        if self.fetch_data_op.get_train_data() is None:
            self.fetch_data_op.execute(train_data, test_data)

        self.model = self.fetch_model_op.get_model()
        self.train_data = self.fetch_data_op.get_train_data()
        self.test_data = self.fetch_data_op.get_test_data()

        if (
            self.model is not None
            and self.train_data is not None
            and self.test_data is not None
        ):
            self._train(epochs, theta, **kwargs)

    @abstractmethod
    def _train(self, *args, **kwargs):
        pass


class ForwardForwardTrainer(BaseForwardForwardTrainer):
    def _train(self, epochs: int, theta: float, **kwargs):
        # Define model
        model = self.model.to(self.device)
        model.epochs = epochs
        batch_size = self.train_data.batch_size

        # TODO: SELECT THE N_CLASSES OUTSIDE THE OPERATION
        label_injector = LabelsInjector(datasets.MNIST.classes)

        progressive_dataset = ProgressiveTrainingDataset(
            (label_injector.inject_train(x, y) for x, y in self.train_data)
        )
        progressive_dataloader = torch.utils.data.DataLoader(
            progressive_dataset, batch_size=batch_size, shuffle=False
        )

        model.train()
        model.progressive_train(progressive_dataloader, theta)

        model.eval()
        correct = 0
        with torch.no_grad():
            for data, target in self.test_data:
                input_data = label_injector.inject_eval(data)
                input_data = input_data.to(self.device)
                target = target.to(self.device)
                _, prob = model.positive_eval(input_data, theta)
                pred = prob.argmax(dim=0)
                correct += pred == target
        if isinstance(correct, torch.Tensor):
            correct = correct.item()
        print(
            "Test set: Accuracy: {}/{} ({:.0f}%)".format(
                correct,
                len(self.test_data.dataset),
                100.0 * correct / len(self.test_data.dataset),
            )
        )


class RecurrentForwardForwardTrainer(BaseForwardForwardTrainer):
    def _train(self, epochs: int, theta: float, **kwargs):
        model = self.model.to(self.device)

        for epoch in range(epochs):
            accumulated_goodness = None
            model.train()
            for j, (data, target) in enumerate(self.train_data):
                # TODO: THE IMAGE SHAPE SHOULD NOT BE DEFINED HERE
                data = data.to(self.device).reshape(-1, 28 * 28)
                target = torch.functional.F.one_hot(
                    target.to(self.device),
                    num_classes=len(datasets.MNIST.classes),
                )
                _, goodness = model.ff_train(data, target, theta)
                if accumulated_goodness is None:
                    accumulated_goodness = goodness
                else:
                    accumulated_goodness[0] += goodness[0]
                    accumulated_goodness[1] += goodness[1]
            goodness_ratio = (
                accumulated_goodness[0] - accumulated_goodness[1]
            ) / abs(max(accumulated_goodness))
            print(f"Epoch {epoch + 1}")
            print(f"Accumulated goodness: {accumulated_goodness}")
            print(f"Goodness ratio: {goodness_ratio}")
            model.eval()
            correct = 0
            with torch.no_grad():
                for data, target in self.test_data:
                    data = data.to(self.device).reshape(-1, 28 * 28)
                    target = target.to(self.device)
                    pred, _ = model.positive_eval(data, theta)
                    correct += pred.eq(target.view_as(pred)).sum().item()
            print(
                f"Test accuracy: {correct} / 10000 ({correct / 10000 * 100}%)"
            )


class NLPForwardForwardTrainer(BaseForwardForwardTrainer):
    def _train(
        self, epochs: int, theta: float, predicted_tokens: int, **kwargs
    ):
        model = self.model.to(self.device)
        self.model.epochs = epochs
        self.model.predicted_tokens = predicted_tokens
        token_num = len(VOCABULARY)
        sequence_len = self.model.seq_len

        for input_data in self.train_data:
            input_data = torch.functional.F.one_hot(
                input_data[0].to(self.device), num_classes=token_num
            ).float()

            accumulated_goodness = model.LM_ff_train(input_data, theta=theta)
            goodness_ratio = (
                accumulated_goodness[0] - accumulated_goodness[1]
            ) / abs(max(accumulated_goodness))
            print("Trained on batch")
            print(f"Accumulated goodness: {accumulated_goodness}")
            print(f"Accumulated goodness ratio: {goodness_ratio}")

        for test_data in self.test_data:
            test_data = torch.functional.F.one_hot(
                test_data[0].to(self.device), num_classes=token_num
            ).float()
            test_data = test_data.reshape(-1, token_num * sequence_len)
            predictions, _ = model.positive_eval(test_data, theta)
            perplexity = compute_perplexity(predictions)
            print(f"Perplexity: {perplexity}")
