import yaml
import os
import shutil

import deepspeed
import torch
from accelerate import Accelerator
from beartype import beartype
from beartype.typing import Tuple, Union, Iterable, Optional
from einops.layers.torch import Rearrange
from peft import get_peft_model, LoraConfig, TaskType
from transformers import (
    AutoModel,
    AutoModelForCausalLM,
    AutoTokenizer,
)

from chatllama.rlhf.config import (
    Config,
    ConfigActor,
    ConfigCritic,
    ConfigReward,
)
from chatllama.rlhf.model_list import (
    hf_models_causal_lm,
    llama_models,
    hf_models,
)

from chatllama.rlhf.model_loader import ModelLoader
from chatllama.rlhf.utils import (
    TrainingStats,
    IgnoreLabelsWrapper,
    LogMessages,
)


ConfigType = Union[ConfigActor, ConfigReward, ConfigCritic, Config]


class BaseModel(torch.nn.Module):
    """Base Model for generic methods implementations for Actor, Critic and
    Reward models. This class is meant to be inherited by the ActorModel,
    CriticModel and RewardModel classes.

    Attributes:
        model: The model used
        tokenizer: The tokenizer used
        config (ConfigActor): Configuration for the model

    Methods:
        load: Load the model from a path
        save: Save the model to a path
        parameters: Return the parameters of the model
        load_tokenizer: Load the tokenizer for the model (staticmethod)
    """
    logger = LogMessages()

    @beartype
    def __init__(self, config: ConfigType) -> None:
        super().__init__()

        # save config
        self.config = config
        
        # if not initialized, initialize the logger
        self.logger = LogMessages(config)

        if not isinstance(config, Config):
            # Actor, Critic or Reward Model initialization

            # initialize the self.model
            if config.model in llama_models:

                # llama is supported only for the actor for NOW
                if not isinstance(config, ConfigActor):
                    raise self.logger.error(
                        ValueError,
                        "LLAMA is supported only for the actor as of now"
                        )

                # llama module might not be present when HF models are used
                from chatllama.llama_model import (
                    load_model,
                    setup_model_parallel,
                )  # noqa

                local_rank, world_size = setup_model_parallel()

                # use load_model_test for testing
                self.model, self.tokenizer = load_model(
                    ckpt_dir=config.model_folder,
                    tokenizer_path=config.tokenizer_path,
                    local_rank=local_rank,
                    world_size=world_size,
                    froze_embeddings=config.froze_embeddings,
                    use_fairscale=config.use_fairscale,
                    max_batch_size=config.batch_size,
                )

            elif config.model in hf_models_causal_lm:

                # load tokenizer
                self.tokenizer = self.load_tokenizer(config)

                # check load 8 bit condition
                if not config.peft_enable:
                    config.load_8bit = False

                # load model
                if isinstance(config, ConfigActor):
                    # load model for the actor
                    self.model = AutoModelForCausalLM.from_pretrained(
                        config.model, load_in_8bit=config.load_8bit
                    )

                elif isinstance(config, ConfigReward) or isinstance(
                    config, ConfigCritic
                ):
                    # load the model for Critic and Reward 
                    # (i.e. without the LM Head)
                    self.model = AutoModel.from_pretrained(
                        config.model, load_in_8bit=config.load_8bit
                    )
                    
                    # define the head for the reward and critic
                    # the head is a ff layer that squash the hidden dimension
                    # to 1 (i.e. the score)
                    head_hidden_size = config.model_head_hidden_size
                    head_dim = self.model.config.hidden_size
                    if config.model.startswith("gpt2"):
                        head_dim = self.model.config.n_embd

                    self.head = torch.nn.Sequential(
                        torch.nn.Linear(head_dim, head_hidden_size),
                        torch.nn.ReLU(),
                        torch.nn.Linear(head_hidden_size, 1),
                        Rearrange("... 1 -> ..."),
                    )
                    
                # apply LoRA with PEFT
                self.is_lora_peft_applied = False
                self.apply_lora_with_peft()

            # move model to device
            self.model.to(config.device)

            # move the head for the reward and critic to device
            if isinstance(config, ConfigReward) or isinstance(
                config, ConfigCritic
            ):
                self.head.to(config.device)

            # load the model from model_folder
            self.load()

        else:
            # ActorCritic initialization

            # if the actor and critic use the same tokenizer is set to True
            self.use_same_tokenizer = False

            # debug flag
            self.debug = config.actor.debug
            
            
    @beartype
    def apply_lora_with_peft(self) -> None:
        """Apply LoRA with PEFT to the model.
        The model is modified in place and the head is not included in the
        PEFTmodel beacause we need to train it as well.
        """
        if self.config.peft_enable:

            # check that the peft config exist
            if (not os.path.exists(self.config.peft_config_path)):
                raise self.logger.error(ValueError,
                    f"PEFT config {self.config.peft_config_path}"
                    f" not found. Can't apply LoRA with PEFT."
                )
                
            # Read the peft config from yaml
            with open(self.config.peft_config_path, "r") as c:
                config_peft = yaml.safe_load(c)

            # define lora config for peft
            peft_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM, **config_peft
            )
            
            # if the model is a reward or critic model
            # needs to be wrapped in a IgnoreLabelsWrapper or lora will pass
            # the labels argument that is not present in the reward and critic
            if isinstance(self.config, ConfigReward) or isinstance(
                self.config, ConfigCritic
            ):
                self.model = IgnoreLabelsWrapper(self.model)

            # create peft model
            self.model = get_peft_model(
                model=self.model,
                peft_config=peft_config,
            )
            
            self.logger.info("LoRA with PEFT applied to the model.")
            
            self.is_lora_peft_applied = True
        

    @beartype
    def load(self) -> None:
        """Load the model from the path, if it is an actor model load only the
        model otherwise if it is an actor or critic model load also the head.

        - For the Actor the saved dictonary contains only one keyword "model"
        corresponding to the whole model weights for the actor.

        - For Critic and Reward the saved dictonary contains two keywords
        "model" and "head".
        """

        if not isinstance(self.config, Config):
            
            # Actor, Critic or Reward Model load()
            if ((not isinstance(self.config, ConfigActor)) and
                (not isinstance(self.config, ConfigReward)) and
                (not isinstance(self.config, ConfigCritic))):
                raise self.logger.error(
                        ValueError,
                        f"Model type not supported: {type(self.config)}"
                    )

            # check if there is a model to load
            path = ModelLoader.check_model_path(
                config=self.config,
                is_checkpoint=False,
                current_epoch=None,
            )

            # if there is a model to load
            if path is not None:
                
                self.logger.info("Loading ...")
                
                # load the model
                model_dict = torch.load(path)
                
                # check model_dict["lora_peft"] and self.is_lora_peft_applied
                # must be the same i.e. lora with peft must be applied
                # to both the model to load and the current model
                if "lora_peft" in model_dict:
                    if model_dict["lora_peft"] != self.is_lora_peft_applied:
                        raise self.logger.error(
                            ValueError,
                            "The model to load is not compatible with the "
                            "current model. The model to load has "
                            f"lora_peft={model_dict['lora_peft']} while "
                            f"the current model has "
                            f"lora_peft={self.is_lora_peft_applied}.",
                        )
                
                if isinstance(self.config, ConfigActor):
                    self.model.load_state_dict(model_dict["model"])

                elif isinstance(self.config, ConfigReward) or isinstance(
                    self.config, ConfigCritic
                ):
                    self.model.load_state_dict(model_dict["model"])
                    self.head.load_state_dict(model_dict["head"])
                    
        else:
            # ActorCritic -- not implemented it relies on the load of the
            # actor and critic
            pass

    @beartype
    def save(self) -> None:
        """Save the model to the path, if it is an actor model save only the
        model otherwise if it is an actor or critic model save also the head.
        In case of ActorCritic model save the actor model as result of RLHF
        in the folder actor_rl instead of actor.save() method that saves it
        in the actor folder.

         - For the Actor the saved dictonary contains only one keyword "model"
        corresponding to the whole model weights for the actor.

        - For Critic and Reward the saved dictonary contains two keywords
        "model" and "head".

        """

        # get the path to save the model
        model_folder, model_name, path = ModelLoader.get_model_path(
            config=self.config,
            is_checkpoint=False,
            current_epoch=None,
        )

        # save the model
        self.logger.info(f"Saving model to {path} ...")
        if isinstance(self.config, ConfigActor):
            # Actor Model Save()
            torch.save(
                {
                    "model": self.model.state_dict(),
                    "lora_peft": self.is_lora_peft_applied,
                },
                path,
            )
        elif isinstance(self.config, ConfigReward) or isinstance(
            self.config, ConfigCritic
        ):
            # Critic or Reward Model save()
            torch.save(
                {
                    "model": self.model.state_dict(),
                    "head": self.head.state_dict(),
                    "lora_peft": self.is_lora_peft_applied,
                },
                path,
            )
        elif isinstance(self.config, Config):
            # ActorCritic save()

            # get the path to save the actor
            model_folder, model_name, path = ModelLoader.get_model_path(
                config=self.config,
                is_checkpoint=False,
            )

            # save the model
            self.logger.info(f"Saving model to {path} ...")
            torch.save(
                {
                    "model": self.actor.model.state_dict(),
                    "lora_peft": self.actor.is_lora_peft_applied,
                },
                path,
            )

            # get the path to save the critic model
            model_folder, model_name, path = ModelLoader.get_model_path(
                config=self.config.critic,
                is_checkpoint=False,
            )

            # save the model
            self.logger.info(f"Saving model to {path} ...")
            torch.save(
                {
                    "model": self.critic.model.state_dict(),
                    "head": self.critic.head.state_dict(),
                    "lora_peft": self.critic.is_lora_peft_applied,  
                },
                path,
            )

    @classmethod
    def load_tokenizer(cls, config: ConfigType):
        """Load the tokenizer from the model name"""
        
        if config.model in hf_models:

            # load the tokenizer from HF
            tokenizer = AutoTokenizer.from_pretrained(
                config.model,
                padding_side="left",
                padding=True,
                truncation=True,
                model_max_length=config.max_sequence_length,
            )

            # add eos token if not present
            if tokenizer.eos_token is None:
                tokenizer.eos_token = "</s>"
                tokenizer.eos_token_id = 2  # OPT eos-token-id

            # add pad token if not present
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
                tokenizer.pad_token_id = tokenizer.eos_token_id

        elif config.model in llama_models:

            if not isinstance(config, ConfigActor):
                raise cls.logger.error(
                    ValueError,
                    "LLaMA models can only be used as actor",
                    )

            # llama module might not be present when HF models are used
            from chatllama.llama_model import (
                load_tokenizer,
            )  # noqa

            tokenizer = load_tokenizer(config.tokenizer_path)
        return tokenizer

    def parameters(self) -> Iterable[torch.nn.Parameter]:
        """Return the parameters of the model"""
        for p in self.model.parameters():
            yield p
        if isinstance(self.config, ConfigReward) or isinstance(
            self.config, ConfigCritic
        ):
            for p in self.head.parameters():
                yield p


class BaseTrainer:
    """Base Class for the trainer of the Actor and Reward models.
    It contains the common methods for the training of the models.
    This class is not meant to be used directly, but to be inherited by
    the ActorTrainer and RewardTrainer classes.

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
        training_stats (List[Dict]): List of dictionaries with the training
            statistics
        model_engine (ModelEngine): Model engine to train the model
            using deepspeed
        accelerator (Accelerator): Accelerator to train the model using
            accelerate by HF.


    Methods:
        train: Train the reward model
        save_checkpoints: Save the checkpoints of the model
        load_checkpoints: Load the checkpoints of the model

    """
    
    logger = LogMessages()

    @beartype
    def __init__(self, config: ConfigType) -> None:
        """Initialize the trainer to be called at the end of the child class
        initialization.
        """

        # save the config
        self.config = config
        
        # if not initialized, initialize the logger
        self.logger = LogMessages(config)

        # initialize trainint stats
        self.trainig_stats = self.setup_training_stats()

        # eps for numerical stability
        self.eps = 1e-8

        # attributes for deepspeed and accelerate
        self.accelerator = None
        self.model_engine = None

        # flags for training
        if isinstance(self.config, Config):
            self.accelerate_enable = self.config.trainer.accelerate_enable
            self.deepspeed_enable = self.config.trainer.deepspeed_enable
            self.deepspeed_config_path = (
                self.config.trainer.deepspeed_config_path
            )  # noqa 501
        else:
            self.accelerate_enable = self.config.accelerate_enable
            self.deepspeed_enable = self.config.deepspeed_enable
            self.deepspeed_config_path = self.config.deepspeed_config_path

        # check consistency of flags
        if self.accelerate_enable and self.deepspeed_enable:
            raise self.logger.error(
                ValueError,
                (   
                    "Both DeepSpeed and Accelerate are enabled" +
                    "Please choose one of them."
                ),
            )

        # check deepspeed config
        if self.deepspeed_enable:
            if self.deepspeed_config_path is None:
                raise self.logger.error(
                    ValueError,
                    "DeepSpeed config path is None, but deepspeed is enabled",
                    )
            if os.path.exists(self.deepspeed_config_path) is False:
                raise self.logger.error(
                    ValueError,
                    f"DeepSpeed config path"
                    f" {self.deepspeed_config_path} "
                    f"does not exist"
                )

    @beartype
    def setup_training_stats(
        self,
    ) -> None:
        """This method initializes the training stats"""
        stats_path = ModelLoader.get_training_stats_path(self.config)
        self.training_stats = TrainingStats(stats_path)

    @beartype
    def append_training_stats(
        self,
        training_loss: Optional[float] = None,
        training_accuracy: Optional[float] = None,
        value_loss: Optional[float] = None,
        validation_loss: Optional[float] = None,
        validation_accuracy: Optional[float] = None,
    ) -> None:
        """
        This method appends the training stats to the training stats list

        Args:
            training_loss (float): Training loss
            training_accuracy (float): Training accuracy
            value_loss (float): Value loss
            validation_loss (float): Validation loss
            validation_accuracy (float): Validation accuracy
        """
        if training_loss is not None:
            self.training_stats.training_loss.append(training_loss)
        elif training_accuracy is not None:
            self.training_stats.training_accuracy.append(training_accuracy)
        elif value_loss is not None:
            self.training_stats.value_loss.append(value_loss)
        elif validation_loss is not None:
            self.training_stats.validation_loss.append(validation_loss)
        elif validation_accuracy is not None:
            self.training_stats.validation_accuracy.append(validation_accuracy)

    @beartype
    def setup_deepspeed(
        self,
    ) -> None:
        """This method initializes the deepspeed engine"""
        deepspeed.init_distributed()

        # initialize deepspeed
        self.model_engine = None
        if self.deepspeed_enable is True:
            (
                self.model_engine,
                self.optimizer,
                self.train_dataloader,
                self.scheduler,
            ) = deepspeed.initialize(
                args=None,
                model=self.model,
                model_parameters=self.model.parameters(),
                optimizer=self.optimizer,
                lr_scheduler=self.scheduler,
                training_data=self.train_dataset,
                config=self.deepspeed_config_path,
            )
            self.logger.info("Training with DeepSpeed")

    @beartype
    def setup_accelerate(
        self,
    ) -> None:
        """This method initializes the accelerator"""

        # initialize accelerate
        self.accelerator = None
        if self.accelerate_enable is True:
            self.accelerator = Accelerator()
            (
                self.model,
                self.optimizer,
                self.train_dataloader,
                self.scheduler,
            ) = self.accelerator.prepare(
                self.model,
                self.optimizer,
                self.train_dataloader,
                self.scheduler,
            )
            self.logger.info("Training with Accelerate")

    @beartype
    def save_checkpoint(
        self,
        current_epoch: int,
        max_epochs: int,
        current_step: Optional[int] = None,
        max_steps: Optional[int] = None,
    ) -> None:
        """Save the checkpoints of the model

        Args:
            current_epoch (int): Current epoch
            current_step (int): Current step
            max_epochs (int): Maximum number of epochs
            max_steps (int): Maximum number of steps
        """

        self.logger.info(
            f"Saving checkpoint for epoch {current_epoch + 1}, "
            f" step {current_step} ..."
        )

        # get the path to save the checkpoint
        model_folder, model_name, path = ModelLoader.get_model_path(
            config=self.config,
            is_checkpoint=True,
            current_epoch=current_epoch,
            current_step=current_step,
            max_epochs=max_epochs,
            max_steps=max_steps,
        )

        # remove the checkpoint if it already exists
        if os.path.exists(path):
            if self.config.deepspeed_enable:
                shutil.rmtree(path)
            else:
                os.remove(path)

        # if deepspeed is enabled
        if self.config.deepspeed_enable:

            # create client state dictonary
            if current_step is None:
                client_state = {
                    "episode": current_epoch,
                }
            else:
                client_state = {
                    "epoch": current_epoch,
                    "step": current_step,
                }

            # save the checkpoint with deepspeed
            self.model_engine.save_checkpoint(path, client_state=client_state)

        else:
            if isinstance(self.config, Config):
                # save actor for ActorCritic Trainer
                # "model" is just for compatibility with load() and save()
                torch.save(
                    {
                        "model": self.model.actor.state_dict(),
                        "critic": self.model.critic.state_dict(),
                        "optimizer": self.scheduler.state_dict(),
                        "scheduler": self.optimizer.state_dict(),
                        "training_stats": self.training_stats,
                        "episode": current_epoch,
                        "lora_peft": self.model.actor.is_lora_peft_applied,
                        "critic_lora_peft": self.model.critic.is_lora_peft_applied, #noqa 501
                    },
                    path,
                )
            else:
                # save the model for other trainers
                torch.save(
                    {
                        "model": self.model.model.state_dict(),
                        "optimizer": self.optimizer.state_dict(),
                        "scheduler": self.scheduler.state_dict(),
                        "training_stats": self.training_stats,
                        "epoch": current_epoch,
                        "step": current_step,
                        "lora_peft": self.model.is_lora_peft_applied,
                    },
                    path,
                )
            self.logger.success(f"Checkpoint saved at {path}")

    @beartype
    def load_checkpoint(
        self,
    ) -> Tuple[int, int]:
        """Load the checkpoints of the model

        Returns:
            Tuple[int, int]: The current epoch and step
                from which you should resume the training
        """

        self.logger.info("Looking for checkpoints...")

        # look for the checkpoints
        path = ModelLoader.check_model_path(
            config=self.config,
            is_checkpoint=True,
            current_epoch=None,
        )

        # check if a checkpoint exists
        if path is not None:
            self.logger.info("Loading ...")

            # if deepspeed is enabled
            if self.config.deepspeed_enable:

                # try to load the checkpoint
                try:
                    _, client_state = self.model_engine.load_checkpoint(path)
                except Exception:
                    self.logger.warning(
                        (
                            "Checkpoint corrupted! " + 
                            "Try to remove the last checkpoint. " +
                            "Now Starting from epoch 0, step 0"
                        )
                    )
                    return 0, 0

                # load epoch and step to resume loops
                if "episode" in client_state:
                    episode = client_state["episode"]
                    return episode, 0
                else:
                    epoch = client_state["epoch"]
                    step = client_state["step"]
                    
                self.logger.success(f"Checkpoint loaded from {path}")
                return epoch, step

            else:

                # try to load the checkpoint
                try:
                    checkpoint = torch.load(path)
                except Exception:
                    self.logger.warning(
                        (
                            "Checkpoint corrupted! " + 
                            "Try to remove the last checkpoint. " +
                            "Now Starting from epoch 0, step 0"
                        )
                    )
                    return 0, 0

                # load optimizer and scheduler state
                self.optimizer.load_state_dict(checkpoint["optimizer"])
                self.scheduler.load_state_dict(checkpoint["scheduler"])

                # load model and epochs
                if isinstance(self.config, Config):
                    # ActorCritic Trainer
                    
                    # first check for lora_peft compatibility
                    if "lora_peft" in checkpoint:
                        if (checkpoint["lora_peft"] !=
                            self.model.actor.is_lora_peft_applied):
                            self.logger.warning(
                                (
                                    "Checkpoint is not compatible with the " +
                                    "current lora_peft setting. " +
                                    "Now Starting from epoch 0, step 0"
                                )
                            )
                            return 0, 0
                    if "critic_lora_peft" in checkpoint:
                        if (checkpoint["critic_lora_peft"] !=
                            self.model.critic.is_lora_peft_applied):
                            self.logger.warning(
                                (
                                    "Checkpoint is not compatible with the " +
                                    "current lora_peft setting. " +
                                    "Now Starting from epoch 0, step 0"
                                )
                            )
                            return 0, 0
                    
                    self.model.actor.load_state_dict(checkpoint["model"])
                    self.model.critic.load_state_dict(checkpoint["critic"])
                    episode = checkpoint["episode"]
                    self.logger.success(f"Checkpoint loaded from {path}")
                    return episode, 0
                else:
                    # Actor and Reward Trainer
                    
                    # first check for lora_peft compatibility
                    if "lora_peft" in checkpoint:
                        if (checkpoint["lora_peft"] !=
                            self.model.is_lora_peft_applied):
                            self.logger.warning(
                                (
                                    "Checkpoint is not compatible with the " +
                                    "current lora_peft setting. " +
                                    "Now Starting from epoch 0, step 0"
                                )
                            )
                            return 0, 0
                    
                    self.model.model.load_state_dict(checkpoint["model"])
                    self.training_stats = checkpoint["training_stats"]
                    epoch = checkpoint["epoch"]
                    step = checkpoint["step"]
                    self.logger.success(f"Checkpoint loaded from {path}")
                    return epoch, step + 1  # return the next episode to train
        return 0, 0
