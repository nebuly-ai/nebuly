import json


import torch
from beartype import beartype
from torch.utils.data import Dataset

from chatllama.rlhf.base_model import BaseModel, BaseTrainer
from chatllama.rlhf.base_model import BaseModel, BaseTrainer
from chatllama.rlhf.config import ConfigReward
from chatllama.rlhf.utils import my_logger


class RewardModel(BaseModel):
    """Model to be trained to predict the reward for RL.
    or to be used as Critic in RL. It is a Language Model with a head
    that predicts the reward (a scalar) for a given sequence of tokens.

    Methods:
        forward: Forward pass of the model (used by the critic)
        get_reward: Get the reward for a given input (used by the reward model)

    """

    def __init__(self, config: ConfigReward) -> None:
        super().__init__(config)

    @beartype
    def forward(
        self, output_sequence: torch.Tensor, output_sequence_mask: torch.Tensor
    ) -> torch.Tensor:
        """Generate the sequence of rewards for the given output sequence
        what is the quality of the output sequence tokens?

        Args:
            output_sequence (torch.Tensor) [batch_size, seq_len]: The sequence
                of tokens to be evaluated
            output_sequence_mask (torch.Tensor) [batch_size, seq_len]: Mask
                for the attention

        Returns:
            torch.Tensor [batch_size, seq_len]: Rewards for the given output
                sequence
        """
        output = self.model.forward(
            output_sequence,
            attention_mask=output_sequence_mask,
        )

        rewards = self.head(output.last_hidden_state)
        if self.config.debug:
            print("RewardModel.forward")
            print("output_sequence.shape", output_sequence.shape)
            print("output_sequence", output_sequence)
            print("reward.shape", rewards.shape)
            print("reward", rewards)
        return rewards

    @beartype
    def get_reward(
        self, output_sequence: torch.Tensor, output_sequence_mask: torch.Tensor
    ) -> torch.Tensor:
        """Get the reward for the given output sequence

        Args:
            output_sequence (torch.Tensor) [batch_size, seq_len]: The
                concatenation of initial input and actor output as tokens
            output_sequence_mask (torch.Tensor) [batch_size, seq_len]: Mask
                for the attention

        Returns:
            torch.Tensor [batch_size]: The reward for the given output sequence
        """
        if output_sequence.shape[1] > self.config.max_sequence_length:
            raise ValueError(
                f"Output sequence is too long: {output_sequence.shape[1]}"
                f" > {self.config.max_sequence_length}"
            )
        rewards = self.forward(output_sequence, output_sequence_mask)
        return rewards[:, -1]


# just to keep namings consistent
CriticModel = RewardModel


class RewardDataset(Dataset):
    """Dataset class for the reward model
    read a json file with the following format:
    [
        {
            "user_input": "...",
            "completion": "...",
            "score": ...
        },
        ...
    ]
    Where:
        user_input: the initial input of the user
        completion: the completion generated by the model
        score: the score given by the user to the completion (or by the LLM)
    """

    def __init__(self, path: str) -> None:
        with open(path, "r") as f:
            self.data = list(json.load(f))

    def __getitem__(self, idx: int):
        user_input = self.data[idx]["user_input"]
        completion = self.data[idx]["completion"]
        if self.data[idx]["score"]:
            score = float(self.data[idx]["score"])
        else:
            score = 2.5

        item = (user_input + completion, score)
        return item

    def __len__(
        self,
    ):
        return len(self.data)


class RewardTrainer(BaseTrainer):
class RewardTrainer(BaseTrainer):
    """Class to train the reward model

    Args:
        config (ConfigModel): Config parameters for the model

    Attributes:
        model (RewardModel): Reward model
        config (ConfigModel): Config parameters for the model
        optimizer (torch.optim): Optimizer for the model
        loss_function (torch.nn): Loss function for the model
        validation_flag (bool): Flag to indicate if the validation dataset
            is available
        train_dataset (RewardDataset): Dataset for training
        validation_dataset (RewardDataset): Dataset for validation
        train_dataloader (DataLoader): Dataloader for training
        validation_dataloader (DataLoader): Dataloader for validation
        scheduler (torch.optim.lr_scheduler): Scheduler for the optimizer

    Methods:
        train: Train the reward model

    Known Issues:
        When using lora and 8 bit precision, the following error is raised:
        - result = F.linear(
            x,
            transpose(self.weight, self.fan_in_fan_out),
            bias=self.bias
            )
        RuntimeError: self and mat2 must have the same dtype
        https://github.com/huggingface/peft/issues/84
    """

    def __init__(self, config: ConfigReward) -> None:

        # load the model
        self.model = RewardModel(config)
        self.model = RewardModel(config)

        # optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=config.lr
            self.model.parameters(), lr=config.lr
        )

        # loss function
        self.loss_function = torch.nn.MSELoss()

        # check validation dataset
        self.validation_flag = False
        if config.validation_dataset_path is not None:
            self.validation_flag = True

        # create dataset
        self.train_dataset = RewardDataset(config.train_dataset_path)
        if self.validation_flag:
            self.eval_dataset = RewardDataset(config.validation_dataset_path)

        # intilize scheduler - learning rate will drop to 10% of the initial
        # value
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=len(self.train_dataset) // config.batch_size,
            T_mult=1,
            eta_min=config.lr * 0.1,
            last_epoch=-1,
        )

        # initialize the super class
        super().__init__(config)

        # dataloader
        self.train_dataloader = self.create_dataloader(
            self.train_dataset, batch_size=config.batch_size
        )
        if self.validation_flag:
            self.validation_dataloader = self.create_dataloader(
                self.eval_dataset, batch_size=config.batch_size
            )

    def train(
        self,
    ) -> None:
        """Train the reward model"""

        my_logger.success("Start Training the Reward Model")

        # get batch size
        if self.deepspeed_enable:
            batch_size = self.model_engine.train_batch_size()
        elif self.accelerate_enable:
            batch_size = (
                self.config.batch_size * self.accelerator.num_processes
            )
        else:
            batch_size = self.config.batch_size

        # get other parameters
        epochs = self.config.epochs
        iteration_per_print = self.config.iteration_per_print
        checkpoint_steps = self.config.checkpoint_steps

        # compute the number of iterations
        n_iter = int(len(self.train_dataset) / batch_size)

        # load checkpoint
        start_epoch, start_step = self.load_checkpoint()

        # counter for the checkpoint
        cnt_checkpoints = 1

        # traing loop
        for epoch in range(start_epoch, epochs):
            self.model.train()
            self.model.train()
            for i, inputs in enumerate(self.train_dataloader):

                # skip the steps if resuming from a checkpoint
                if i < start_step:
                    continue

                # get tensor for forward and loss computation
                with torch.no_grad():

                    # get the inputs
                    input_text = inputs[0]
                    score = inputs[1]

                    # tokenize the input
                    if self.accelerate_enable:
                        input_tokens = self.model.module.tokenizer(
                            input_text,
                            return_tensors="pt",
                            truncation=True,
                            padding=True,
                        )
                    else:
                        input_tokens = self.model.tokenizer(
                            input_text,
                            return_tensors="pt",
                            truncation=True,
                            padding=True,
                        )

                    # assign the input and output
                    output = torch.as_tensor(score, dtype=torch.float16)
                    input_ids = input_tokens["input_ids"]
                    input_mask = input_tokens["attention_mask"]

                    # move to device
                    if not self.config.load_8bit:
                        input_ids = input_ids.to(self.device)
                        input_mask = input_mask.to(self.device)
                        output = output.to(self.device)

                # forward pass
                if self.config.deepspeed_enable:
                    # deepspeed
                    est_output = self.model_engine(
                        input_ids,
                        input_mask,
                    )[:, -1]
                elif self.accelerate_enable:
                    # accelerate
                    est_output = self.model(
                        input_ids,
                        input_mask,
                    )[:, -1]
                else:
                    # Pytorch
                    with torch.autocast(
                        device_type=self.config.device_type,
                        dtype=torch.float16,
                    ):
                        est_output = self.model(
                            input_ids,
                            input_mask,
                        )[:, -1]

                # compute the loss
                if self.deepspeed_enable or self.accelerate_enable:
                    # DeepSpeed and Accelerate
                    loss = self.loss_function(est_output, output)

                else:
                    # if mixed precision with pytorch use autocast
                    with torch.autocast(
                        device_type=self.config.device_type,
                        dtype=torch.float16,
                    ):
                        loss = self.loss_function(est_output, output)

                # backward pass
                if self.deepspeed_enable:
                    # DeepSpeed
                    self.model_engine.backward(loss)
                    self.model_engine.step()
                elif self.accelerate_enable:
                    # Accelerate
                    self.optimizer.zero_grad()
                    self.accelerator.backward(loss)
                    self.optimizer.step()
                    self.scheduler.step()
                else:
                    # Pytorch
                    self.optimizer.zero_grad()
                    if self.config.load_8bit:
                        # 8 bit from HF
                        loss.backward()
                        self.optimizer.step()
                    else:
                        # Pytorch Mixed Precision
                        self.scaler.scale(loss).backward()
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    self.scheduler.step()

                # save training stats
                self.append_training_stats(training_loss=loss.detach().item())

                # print progress
                if i % iteration_per_print == 0:
                    my_logger.info(
                        f"Epoch: {epoch+1}/{epochs}, "
                        f"Iteration: {i+1}/{n_iter}, "
                        f"Training Loss: {loss.item()}"
                    )
                    printed_est_output = [
                        round(float(x), 1) for x in est_output.cpu().tolist()
                    ]
                    my_logger.info(
                        f"prediction {printed_est_output} "
                        f"target {score.cpu().tolist()}"
                    )

                # checkpoints saving
                if cnt_checkpoints % checkpoint_steps == 0:
                    self.save_checkpoint(
                        current_epoch = epoch,
                        max_epochs = epochs,
                        current_step = i,
                        max_steps = n_iter,
                        )
                    cnt_checkpoints = 1
                else:
                    cnt_checkpoints += 1

            # Validation
            if self.validation_flag:
                self.model.eval()
                self.model.eval()
                with torch.no_grad():
                    for i, (text, score) in enumerate(
                        self.validation_dataloader
                    ):

                        # tokenize inputs
                        input_tokens = self.model.tokenizer(
                        input_tokens = self.model.tokenizer(
                            text, return_tensors="pt", padding=True
                        )
                        input_tokens = input_tokens.to(self.device)
                        # TODO: check on the length of the input tokens if
                        # they are too many it can create problems
                        output = torch.tensor(score, dtype=torch.float32).to(
                            self.device
                        )

                        # forward pass
                        est_output = self.model.get_reward(
                        est_output = self.model.get_reward(
                            input_tokens["input_ids"],
                            input_tokens["attention_mask"],
                        )

                        # compute loss
                        loss = self.loss_function(est_output, output)
                        self.append_training_stats(validation_loss=loss.item())
                        self.append_training_stats(validation_loss=loss.item())

                        # print progress
                        if i % iteration_per_print == 0:
                            my_logger.info(
                                f"Epoch: {epoch+1}/{epochs}, "
                                f"Iteration: {i+1}/{n_iter}, "
                                f"Validation Loss: {loss.item()}"
                            )
            # reset start_step after training is resumed
            start_step = 0

        # save the model at the end of the training
        if self.accelerate_enable:
            self.model.module.save()
        else:
            self.model.save()
        my_logger.success("Training is finished")
