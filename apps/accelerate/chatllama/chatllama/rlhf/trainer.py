import json
import os
import random
from collections import deque, namedtuple

import torch
from beartype import beartype
from beartype.typing import Deque, List, Tuple, Union
from torch.utils.data import DataLoader, Dataset

from chatllama.rlhf.actor import ActorModel
from chatllama.rlhf.base_model import BaseModel, BaseTrainer
from chatllama.rlhf.config import (
    Config,
    ConfigActor,
    ConfigCritic,
    ConfigReward,
)
from chatllama.rlhf.model_list import hf_models
from chatllama.rlhf.model_loader import ModelLoader
from chatllama.rlhf.reward import RewardModel, CriticModel
from chatllama.rlhf.utils import ConversationLog


"""
train()
┌─────────────────────────────┐
│                             │◄─────────────────────────┐
│                             │                          │
│      ┌─────────────┐        │                          │
│      │ user input  │        │                          │ learn()
│      └─────┬───────┘        │             ┌────────────┴─────────────┐
│            │                │             │                          │
│            │                │             │       ┌────────┐         │
│            │                │             │   ┌───│ Update │──┐      │
│            │                │             │   │   └────▲───┘  │      │
│   ┌────────▼────────────┐   │             │   │        │      │      │
│   │  Actor (LLM Model)  │   │             │   │     ┌──┴───┐  │      │
│   └────────┬────────────┘   │             │   │     │ PPO  │  │      │
│            │                │             │   │     └▲────▲┘  │      │
│            │                │             │   │      │    │   │      │
│            │                │             │   │      │    │   │      │
│    ┌───────▼──────┐         │             │ ┌─▼──────┴┐ ┌─┴───▼──┐   │
│    │ Reward Model │         │             │ │  Actor  │ │ Critic │   │
│    └──────────────┘         │             │ └─────────┘ └────────┘   │
│                             │             │                          │
│                             │ x Episodes  └─────────────▲────────────┘
└───────────────┬─────────────┘                           │   x Epochs
                │ store N Examples per Timestep           │  
         ┌──────▼──────┐                                  │
         │             │                                  │
         │  Memories   ├──────────────────────────────────┘
         │             │ (update timesteps x N Examples)
         └─────────────┘
"""  # noqa W291


def change_tokenization(tokens, tokenizer1, tokenizer2):
    """Change the tokenizer of the tokens

    Args:
        tokens (torch.Tensor): Tokens to be changed
        tokenizer1 (transformers.PreTrainedTokenizer): Tokenizer to be changed
        tokenizer2 (transformers.PreTrainedTokenizer): Tokenizer to be
            changed to

    Returns:
        encoded_tokens: Encoded tokens
    """

    # decode tokens
    with torch.no_grad():
        decoded_tokens = [
            tokenizer1.decode(token) for i, token in enumerate(tokens)
        ]

        # remove all the pad tokens
        decoded_tokens = [
            token.replace(tokenizer1.pad_token, "") for token in decoded_tokens
        ]

        # remove all the eos tokens
        decoded_tokens = [
            token.replace(tokenizer1.eos_token, "") for token in decoded_tokens
        ]

        # encode the actions with critic tokenizer
        encoded_tokens = tokenizer2(
            decoded_tokens,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )

    return encoded_tokens


ConfigType = Union[ConfigActor, ConfigReward, ConfigCritic]


@beartype
def check_model_family(config1: ConfigType, config2: ConfigType) -> bool:
    """Check if the model family is the same for the two configs
    the model family is specified in the config.model

    Args:
        config1 (ConfigType): First config
        config2 (ConfigType): Second config

    Returns:
        bool: True if the model family is the same, False otherwise
    """

    # check if both are an hugging face models
    if (config1.model in hf_models) and (config2.model in hf_models):

        # if there is a "/" remove it from the name
        model_name1 = config1.model
        model_name2 = config2.model
        if "/" in model_name1:
            model_name1 = model_name1.split("/")[1]
        if "/" in model_name2:
            model_name2 = model_name2.split("/")[1]

        # check if the model family is the same
        return model_name1.split("-")[0] == model_name2.split("-")[0]

    # check if both are not an hugging face models
    elif (config1.model not in hf_models) and (config2.model not in hf_models):

        # for now they could be only LLaMA models
        return True
    else:
        return False


class ActorCritic(BaseModel):
    """Actor Critic class stores both the actor and the critic models
    and it generates values and action for given sequences during the training
    of the actor.

    Attributes:
        actor (ActorModel): Actor model
        critic (CriticModel): Critic model

    Methods:
        forward: given a sequence returns action logits and values (used
            to evaluate the actor during training)
        generate: given a sequence returns action, action logits, values
            sequences and sequences masks (used to generate new sequences
            during acting phase)
    """

    def __init__(self, config: Config) -> None:
        super().__init__(config)

        # load the actor
        self.actor = ActorModel(config.actor)

        # check if critic must be initialized from reward model
        ModelLoader.init_critic_from_reward(config.critic)

        # now load the critic
        self.critic = CriticModel(config.critic)

    @beartype
    def forward(
        self,
        sequences_actor: torch.Tensor,
        sequences_mask_actor: torch.Tensor,
        sequences_critic: torch.Tensor,
        sequences_mask_critic: torch.Tensor,
        action_len_actor: int,
        action_len_critic: int,
    ) -> Tuple:
        """Given the whole sequences, use the actor forward to get the logits
            for each token in the sequence and the critic forward to get the
            values for each generation step.

        Args:
            sequences_actor (torch.Tensor): Sequences composed of
                [states, actions] for the actor
            sequence_mask_actor (torch.Tensor): Mask for the sequences
                of the actor
            sequences_critic (torch.Tensor): Sequences composed of
                [states, actions] for the critic
            sequences_mask_critic (torch.Tensor): Mask for the sequences
                of the critic
            action_len_actor (int): Length of the actions in the sequences
                for the actor
            action_len_critic (int): Length of the actions in the sequences
                for the critic

        Returns:
            action_logits (torch.Tensor): Logits for the actions in the
                sequences
            values (torch.Tensor): Values for the actions in the sequences
        """

        # use a single forward on the whole sequence
        # to get pi(y | x) and ignore predicted output
        actions_logits = self.actor.forward(
            sequences_actor, sequences_mask_actor
        )

        # use the critic forward to get the values for the actions
        values = self.critic.forward(sequences_critic, sequences_mask_critic)

        # return only logits and values for the actions taken
        real_actions_logits = actions_logits[:, -action_len_actor:, :]
        real_values = values[:, -action_len_critic:]

        if self.debug:
            print("ActorCritic.forward")
            print("action_len_actor", action_len_actor)
            print("action_len_critic", action_len_critic)
            print("sequences_actor.shape", sequences_actor.shape)
            print("sequences_actor", sequences_actor)
            print("sequences_critic.shape", sequences_critic.shape)
            print("sequences_critic", sequences_critic)
            print("real_action_logits.shape", actions_logits.shape)
            print("real_action_logits", actions_logits)
            print("real_values.shape", values.shape)
            print("real_values", values)

        return (
            real_actions_logits,
            real_values,
        )

    @torch.no_grad()
    @beartype
    def generate(
        self,
        states_actor: torch.Tensor,
        states_mask_actor: torch.Tensor,
        states_critic: torch.Tensor,
    ) -> Tuple:
        """Generate actions, actions_logits, values and sequences from states

        Args:
            states_actor (torch.Tensor): States for the actor
            states_mask_actor (torch.Tensor): Mask for the states for the
                actor
            states_critic (torch.Tensor): States for the critic

        Returns:
            actions (torch.Tensor): Actions generated from the states
            actions_logits (torch.Tensor): Logits for the actions generated
                from the states (i.e. pi(y | x))
            values (torch.Tensor): Values generated by the critic model
                for the actions generated by the actor (i.e. V(x))
            sequences (torch.Tensor): Sequences generated from the states
                as [states, actions]
        """

        # generate action sequence from the actor
        actions, sequences_actor = self.actor.generate(
            states_actor, states_mask_actor
        )

        # create mask for the actor sequences
        sequences_mask_actor = (
            (sequences_actor != self.actor.tokenizer.pad_token_id)
            .to(sequences_actor.device)
            .long()
            .detach()
        )

        # get the length of the actions
        action_len_actor = actions.shape[1]

        # check if different encoding is needed for the critic
        if self.use_same_tokenizer:
            sequences_critic = sequences_actor
            sequences_mask_critic = sequences_mask_actor
            action_len_critic = action_len_actor
        else:
            encoded_critic = change_tokenization(
                sequences_actor,
                self.actor.tokenizer,
                self.critic.tokenizer,
            )
            # split the encoded_critic in tokens and maks
            sequences_critic = encoded_critic["input_ids"].to(
                sequences_actor.device,
            )
            sequences_mask_critic = (
                encoded_critic["attention_mask"]
                .to(sequences_actor.device)
                .long()
                .detach()
            )

            # compute len of actions for the critic tokenizer
            action_len_critic = states_critic.shape[1]

        # generate actions_logits and values
        actions_logits, values = self.forward(
            sequences_actor,
            sequences_mask_actor,
            sequences_critic,
            sequences_mask_critic,
            action_len_actor,
            action_len_critic,
        )
        if self.debug:
            print("ActorCritic.generate")
            print("actions shape", actions.shape)
            print("actions", actions)
            print("sequence shape", sequences_actor.shape)
            print("sequence", sequences_actor)
            print("actions_logits shape", actions_logits.shape)
            print("actions_logits", actions_logits)
            print("values shape", values.shape)
            print("values", values)

        return (
            actions,
            actions_logits,
            values,
            sequences_actor,
            sequences_mask_actor,
            sequences_critic,
            sequences_mask_critic,
            action_len_actor,
            action_len_critic,
        )


# structure to store the data for each experience
Memory = namedtuple(
    "Memory",
    [
        "states_actor",
        "actions",
        "values",
        "rewards",
        "actions_log_probs",
        "sequences_actor",
        "sequences_mask_actor",
        "sequences_critic",
        "sequences_mask_critic",
        "action_len_actor",
        "action_len_critic",
    ],
)


class ExperienceDataset(Dataset):
    """Dataset to train the actor-critic models"""

    def __init__(
        self,
        memories: Deque[Memory],
        device: torch.device,
    ) -> None:
        super().__init__()
        self.data = list(memories)
        self.device = device

    def __len__(
        self,
    ) -> int:
        return len(self.data)

    def __getitem__(self, idx) -> Tuple:
        # return the idx-th memory element as a tuple of tensors on the device
        item = (
            self.data[idx].states_actor.to(self.device),
            self.data[idx].actions.to(self.device),
            self.data[idx].values.to(self.device),
            self.data[idx].rewards.to(self.device),
            self.data[idx].actions_log_probs.to(self.device),
            self.data[idx].sequences_actor.to(self.device),
            self.data[idx].sequences_mask_actor.to(self.device),
            self.data[idx].sequences_critic.to(self.device),
            self.data[idx].sequences_mask_critic.to(self.device),
            int(self.data[idx].action_len_actor),
            int(self.data[idx].action_len_critic),
        )
        return item


class ExamplesSampler:
    """Store the prompt to be sampled to generate the examples
    read a json file with the following format:
    [
        {
            "user_input" : "",
        } ,
        ...
    ]
    Where:
        user_input: is the input of the user or directly the input of the user
            with the memory preappended (i.e. user_input + memory)
    """

    def __init__(
        self,
        path: str,
    ) -> None:
        self.path = path
        with open(path, "r") as f:
            data = json.load(f)
        self.data = [d["user_input"] for d in data]

    def sample(self, n: int) -> List:
        """Sample n examples from the data

        Args:
            n (int): Number of examples to sample
        """
        return random.sample(self.data, n)
    
    
class CustomOptimizer(torch.optim.Optimizer):
    
    def __init__(self, params_actor, params_critic, lr_actor, lr_critic):
        self.actor = torch.optim.AdamW(params_actor, lr=lr_actor)
        self.critic = torch.optim.AdamW(params_critic, lr=lr_critic)
        
    def step(self):
        self.actor.step()
        self.critic.step()
        
    def zero_grad(self):
        self.actor.zero_grad()
        self.critic.zero_grad()
    

class RLTrainer(BaseTrainer):
    """Train the actor-critic model using RL

    Attributes:
        config (Config): Configuration of the trainer
        debug (bool): Debug mode
        actorcritic (ActorCritic): Actor-critic model
        actor_optim (torch.optim): Optimizer for the actor
        critic_optim (torch.optim): Optimizer for the critic
        actor_scheduler (torch.optim.lr_scheduler): Scheduler for the actor
        critic_scheduler (torch.optim.lr_scheduler): Scheduler for the critic
        reward (RewardModel): Reward model
        training_stats (TrainingStats): Class to store training stats
        conversation_log (ConversationLog): Class to store the conversation
        examples_sampler (ExamplesSampler): Class to sample examples
        eps (float): small epsilon to avoid division by zero

    Methods:
        train: the training loop that calls the learn function after generating
            the experiences.
        learn: Learn from a batch of experiences and update the actor and the
            critic model.
        load_checkpoint: Load the checkpoint of the actor-critic model
        save_checkpoint: Save the checkpoint of the actor-critic model
    """

    def __init__(
        self,
        config: Config,
    ) -> None:
        
        super().__init__(config)

        # set debug mode
        self.debug = config.trainer.debug

        # initialize actor-critic
        self.actorcritic = ActorCritic(config)

        # initialize actor optimizer
        self.optimizer = CustomOptimizer(
            self.actorcritic.actor.parameters(),
            self.actorcritic.critic.parameters(),
            config.trainer.actor_lr,
            config.trainer.critic_lr,
        )

        # scheduler (defined in the learn() method (i need dataset size))
        self.scheduler = None

        # initialize reward model
        self.reward = RewardModel(config.reward)

        # initialize class to store conversations logs
        model_folder, _, _ = ModelLoader.get_model_path(
            config,
            is_checkpoint=True,
        )
        path = os.path.join(model_folder, "conversations_log.json")
        self.conversation_log = ConversationLog(path)

        # initialize examples sampler
        self.example_sampler = ExamplesSampler(config.trainer.examples_path)

        # check if actor and critic use the same tokenizer
        self.actorcritic.use_same_tokenizer = check_model_family(
            config.actor, config.critic
        )

        # check if actor and reward use the same tokenizer
        self.use_same_tokenizer = check_model_family(
            config.actor, config.reward
        )

    @beartype
    def learn(self, memories: Deque[Memory]) -> None:
        """Train the agent-critic model using RL:
        - for each batch of episodes, compute action logits and values
        - then compare action logits probs with memories one and values with
            rewards to compute the PPO loss and update the actor-critic model
        """
        print("Start to Learn...")

        # get parameters
        epochs = self.config.trainer.epochs
        actor_eps_clip = self.config.trainer.actor_eps_clip
        critic_eps_clip = self.config.trainer.critic_eps_clip
        beta_s = self.config.trainer.beta_s
        batch_size = self.config.trainer.batch_size
        device = self.config.trainer.device

        # create dataset from memories
        self.train_dataset = ExperienceDataset(memories, device)
        self.train_dataloader = DataLoader(self.train_dataset, batch_size=batch_size)

        # initialize scheduler 
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer.actor,
            T_0=len(self.train_dataset) // batch_size,
            T_mult=1,
            eta_min=self.config.trainer.actor_lr * 0.1,
        )

        # setup deepspeed
        self.setup_deepspeed()
        
        # setup accelerate
        self.setup_accelerate()

        # train agent-critic
        self.actorcritic.train()
        for epoch in range(epochs):
            for k, (
                states_actor,
                old_actions,
                old_values,
                rewards,
                old_actions_log_probs,
                sequences_actor,
                sequences_mask_actor,
                sequences_critic,
                sequences_mask_critic,
                action_len_actor,
                action_len_critic,
            ) in enumerate(self.train_dataloader):

                if self.debug:
                    print(
                        f"#########################################"
                        f" batch from memories {k} \n "
                        f"#########################################"
                        f"states_actor {states_actor.shape} \n"
                        f"old_actions {old_actions.shape} \n"
                        f"old_values {old_values.shape} \n"
                        f"rewards {rewards.shape} \n"
                        f"old_actions_log_probs "
                        f"{old_actions_log_probs.shape}\n"
                        f"sequences_actor {sequences_actor.shape} \n"
                        f"sequences_mask_actor "
                        f"{sequences_mask_actor.shape} \n"
                        f"sequences_critic {sequences_critic.shape} \n"
                        f"sequences_mask_critic "
                        f"{sequences_mask_critic.shape} \n"
                        f"action_len_actor {action_len_actor} \n"
                        f"action_len_critic {action_len_critic} \n"
                        f"#########################################"
                    )

                # get actor critic new probabilities and values
                actions_logits, values = self.actorcritic.forward(
                    sequences_actor,
                    sequences_mask_actor,
                    sequences_critic,
                    sequences_mask_critic,
                    action_len_actor.item(),
                    action_len_critic.item(),
                )

                # get action log prob
                actions_prob = (
                    torch.softmax(actions_logits, dim=-1).max(dim=-1).values
                )
                actions_log_prob = torch.log(actions_prob + self.eps)

                # compute entropy
                entropies = (actions_prob * actions_log_prob).sum(dim=-1)

                # compute KL divergence
                kl_div_loss = (
                    (actions_prob * (old_actions_log_probs - actions_log_prob))
                    .sum(dim=-1)
                    .mean()
                )

                # compute ratios
                ratios = (actions_log_prob - old_actions_log_probs).exp()

                # compute PPO loss
                if check_model_family(self.config.actor, self.config.critic):
                    # compute discounted rewards as in TRL
                    gamma = self.config.trainer.gamma_discounted
                    discounted_rewards = torch.zeros_like(old_values)
                    for i in range(discounted_rewards.shape[1]):
                        for j in range(i, discounted_rewards.shape[1]):
                            discounted_rewards[:, i] += (
                                gamma ** (j - i) * rewards[:, j]
                            )

                    advantages = (
                        discounted_rewards - old_values
                    )  # TRL has opposite sign for old values
                    advantages = (advantages - advantages.mean(dim=-1)) / (
                        advantages.std() + self.eps
                    )

                    surr1 = advantages * ratios
                else:
                    advantages = rewards - old_values[:, -1]
                    surr1 = advantages * ratios

                surr2 = (
                    torch.clamp(ratios, 1 - actor_eps_clip, 1 + actor_eps_clip)
                    * advantages
                )

                policy_loss = -torch.min(surr1, surr2) - beta_s * entropies
                policy_loss = policy_loss.mean()
                policy_loss = policy_loss + kl_div_loss

                # check if loss item is NaN
                if torch.isnan(policy_loss):
                    raise ValueError("Policy Loss is nan")

                # compute value loss
                # the loss is the distance between the rewards and the values
                # I want this distance to be small so that values are
                # representative of the rewards, for this reason i took the
                # maximum between the two.
                # The clip is limiting the slew-rate of values_loss_clipped
                value_loss_clipped = old_values + (values - old_values).clamp(
                    -critic_eps_clip, critic_eps_clip
                )
                value_loss1 = (value_loss_clipped - rewards) ** 2
                value_loss2 = (values - rewards) ** 2
                value_loss = torch.max(value_loss1, value_loss2).mean()

                if torch.isnan(value_loss):
                    raise ValueError("Value loss is nan")
                
                # Sum the two losses
                loss = policy_loss + value_loss

                # backward pass
                if self.deepspeed_enable:
                    # DeepSpeed backward pass
                    self.model_engine.backward(loss)
                    self.model_engine.step()
                    
                elif self.accelerate_enable:
                    # Accelerate backward pass
                    self.optimizer.zero_grad()
                    self.accelerator.backward(loss)
                    self.optimizer.step()
                    self.scheduler.step()
                else:
                    # PyTorch backward pass
                    self.optimizer.zero_grad()
                    value_loss.backward()
                    self.optimizer.step()
                    self.scheduler.step()
                    
                # append the losses to the training stats
                self.append_training_stats(
                    training_loss = loss.detach().cpu().item(),
                    value_loss = value_loss.detach().cpu().item(),
                )

                # print iteration info
                print(
                    f"Epoch {epoch+1}/{epochs}",
                    f"Step {k+1}/{int(len(self.train_dataloader) / batch_size)}",
                    f"Loss {loss.detach().cpu().item():.4f}",
                    f"Value Loss {value_loss.detach().cpu().item():.4f}",
                )

        self.actorcritic.eval()
        print("End Learning")

    def train(
        self,
    ) -> None:

        print("Start RL Training")

        # initialize settings
        num_episodes = self.config.trainer.num_episodes
        max_timesteps = self.config.trainer.max_timesteps
        num_examples = self.config.trainer.num_examples
        update_timesteps = self.config.trainer.update_timesteps
        batch_size = self.config.trainer.batch_size
        checkpoint_steps = self.config.trainer.checkpoint_steps
        device = self.config.trainer.device

        # number of elements that the memories should contain when learning
        number_of_memories_per_learn_iteration = (
            num_examples * update_timesteps
        )

        # the number of memories must be a multiple of the batch size
        assert (
            number_of_memories_per_learn_iteration % batch_size == 0
        ), "The number of memories must be a multiple of the batch size"

        # the total number of timesteps done in the train() are
        total_number_of_timesteps = num_episodes * max_timesteps

        # the total timesteps done should be a multiple of the update timesteps
        assert total_number_of_timesteps % update_timesteps == 0, (
            "The number of timesteps (num_episodes*max_timesteps)"
            "must be a multiple of the update_timesteps"
        )

        # initialize memories
        memories = deque([])

        # load checkpoint
        start_episode, _ = self.load_checkpoint()

        # if it is a new training from the start clear the conversation log
        if start_episode == 0:
            self.conversation_log.clear()

        # initialize counters
        cnt_timesteps = 0
        cnt_learn_iter = 0

        # loop over episodes and timesteps
        self.actorcritic.eval()
        for episode in range(start_episode, num_episodes):
            for timestep in range(max_timesteps):

                # print the iteration info
                print(
                    f"Episode: {episode + 1}/{num_episodes}, "
                    f"Timestep: {timestep + 1}/{max_timesteps}",
                    f"Learning Cnt: {cnt_timesteps + 1}/{update_timesteps}",
                )

                # counter used to count timesteps into memory
                cnt_timesteps += 1

                # sample num_examples examples from  example dataset
                inputs = self.example_sampler.sample(num_examples)

                # tokenize examples for the actor
                tok_inputs_act = self.actorcritic.actor.tokenizer(
                    inputs, padding=True, return_tensors="pt", truncation=True
                )

                # states are [batch_size, seq_len_of_states]
                states_actor = tok_inputs_act["input_ids"].to(device)
                states_mask_actor = tok_inputs_act["attention_mask"].to(device)

                # tokenize examples for the critic
                tok_inputs_crt = self.actorcritic.critic.tokenizer(
                    inputs, padding=True, return_tensors="pt", truncation=True
                )

                # states are [batch_size, seq_len_of_states]
                states_critic = tok_inputs_crt["input_ids"].to(device)

                # generate sequences of actions and values
                (
                    actions,
                    actions_logits,
                    values,
                    sequences_actor,
                    sequences_mask_actor,
                    sequences_critic,
                    sequences_mask_critic,
                    action_len_actor,
                    action_len_critic,
                ) = self.actorcritic.generate(
                    states_actor, states_mask_actor, states_critic
                )

                # compute action log probs
                action_prob = (
                    torch.softmax(actions_logits, dim=-1).max(dim=-1).values
                )
                actions_log_probs = torch.log(action_prob + self.eps)

                # get tokenized sequence for the reward models
                if self.use_same_tokenizer:
                    reward_sequence = sequences_actor
                    reward_mask = sequences_mask_actor
                elif check_model_family(
                    self.config.critic, self.config.reward
                ):
                    reward_sequence = sequences_critic
                    reward_mask = sequences_mask_critic
                else:
                    tokenized_responses = change_tokenization(
                        sequences_actor,
                        self.actorcritic.actor.tokenizer,
                        self.reward.tokenizer,
                    )
                    # get tokens and mask
                    reward_sequence = tokenized_responses["input_ids"].to(
                        device
                    )
                    reward_mask = tokenized_responses["attention_mask"].to(
                        device
                    )

                # compute rewards
                rewards = self.reward.forward(
                    reward_sequence,
                    reward_mask,
                )

                rewards = rewards[:, -action_len_critic:]
                reward = rewards[:, -1]

                # store memories of the episode / timestep
                for i in range(states_actor.shape[0]):
                    memories.append(
                        Memory(
                            states_actor[i, :].detach().cpu(),
                            actions[i, :].detach().cpu(),
                            values[i, :].detach().cpu(),
                            rewards[i, :].detach().cpu(),
                            actions_log_probs[i, :].detach().cpu(),
                            sequences_actor[i, :].detach().cpu(),
                            sequences_mask_actor[i, :].detach().cpu(),
                            sequences_critic[i, :].detach().cpu(),
                            sequences_mask_critic[i, :].detach().cpu(),
                            int(action_len_actor),
                            int(action_len_critic),
                        )
                    )

                # decode completions to be logged in the conversation log
                completions = [
                    self.actorcritic.actor.tokenizer.decode(action)
                    for action in actions
                ]
                # remove pad tokens from completions
                completions = [
                    c.replace(self.actorcritic.actor.tokenizer.pad_token, "")
                    for c in completions
                ]
                # remove eos tokens from completions
                completions = [
                    c.replace(self.actorcritic.actor.tokenizer.eos_token, "")
                    for c in completions
                ]
                # strange i need to force this?
                completions = [c.replace("<pad>", "") for c in completions]

                # log the memories in the conversation log
                for i in range(states_actor.shape[0]):
                    self.conversation_log.append(
                        inputs[i],
                        completions[i],
                        reward[i].detach().cpu().item(),
                        cnt_learn_iter,
                    )

                # learn from memories
                if (cnt_timesteps % update_timesteps == 0) and (
                    cnt_timesteps != 0
                ):
                    print("len memories", len(memories))
                    # self.conversation_log.show(cnt_learn_iter)
                    self.learn(memories)
                    mean_reward = sum([m.rewards[-1] for m in memories]) / len(
                        memories
                    )
                    print(f"Mean Reward: {mean_reward}")
                    memories.clear()
                    cnt_timesteps = 0
                    cnt_learn_iter += 1
                    self.conversation_log.save()

            # save checkpoints
            if (episode % checkpoint_steps == 0) and (episode != 0):
                self.save_checkpoint(
                    current_episode=episode, max_episode=num_episodes
                )
                self.conversation_log.save()

        # save the models
        self.actorcritic.save()
        print("End RL Training")
