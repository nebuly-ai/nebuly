import json
import os

import numpy as np
import pandas as pd
from beartype.typing import Dict, List, Union
from datasets import load_dataset
from chatllama.rlhf.config import Config, ConfigActor, ConfigReward
from chatllama.rlhf.reward import RewardModel, CriticModel
from chatllama.rlhf.actor import ActorModel


ConfigType = Union[Config, ConfigActor, ConfigReward]


class BaseDataset:
    def __init__(
        self,
    ) -> None:
        pass

    @staticmethod
    def sort_conversation(
        conversations: List[Dict],
        only_input: bool = False,
        reverse: bool = True,
        shuffle: bool = True,
    ) -> List[Dict]:
        """Sort the conversations by length of user_input + completion
        or by length of user_input only

        Args:
            conversations (List[Dict]): list of conversations
            only_input (bool, optional): sort by length of user_input only.
                Defaults to False.
            reverse (bool, optional): sort in descending order.
                Defaults to True.
            shuffle (bool, optional): shuffle the dataset leaving only the
                first 100 samples sorted. Defaults to True.

        Returns:
            List[Dict]: sorted list of conversations
        """

        # define the sorting function
        if only_input is True:

            def sort_fun(x):
                return len(x["user_input"])

        else:

            def sort_fun(x):
                return len(x["user_input"]) + len(x["completion"])

        # sort
        conversations = sorted(
            conversations,
            key=sort_fun,
            reverse=reverse,
        )

        # shuffle
        if shuffle is True:
            conversations = (
                conversations[:10]
                + np.random.choice(
                    conversations[10:],
                    size=len(conversations[10:]),
                    replace=False,
                ).tolist()
            )

        return conversations

    @staticmethod
    def take_n_samples(
        conversations: List[Dict],
        n: int,
    ) -> List[Dict]:
        """Take N samples from the dataset

        Args:
            conversations (List[Dict]): list of conversations
            n (int): number of samples to take randomly

        Returns:
            List[Dict]: list of N samples
        """

        # sample N number of index from 0 to len(conversations)
        indexes = np.random.choice(len(conversations), size=n, replace=False)
        # take the samples
        conversations = [conversations[i] for i in indexes]
        return conversations

    @staticmethod
    def clean_dataset(config: ConfigType):
        """Clean the datasets by removing too long examples
        The Reward Dataset constraints are:
        - user_input + completion < Reward model max sequence length
        The Actor Dataset constraints are:
        - user_input + completion < Actor model max sequence length
        The RLHF Training Dataset constraints are:
        - user_input + min_completion < Actor model max sequence length
        - user_input + min_completion < Critic model max sequence length
        - user_input + min_completion < Reward model max sequence length

        Args:
            config (Config): config object
        """

        if isinstance(config, Config):
            print("Start cleaning the dataset for RLHF")
            # constraints
            r_model_max_seq_len = config.reward.max_sequence_length
            a_model_max_seq_len = config.actor.max_sequence_length
            c_model_max_seq_len = config.critic.max_sequence_length
            min_completion = config.actor.min_tokens
            # dataset
            dataset_path = config.trainer.examples_path
            # tokenizers
            r_tokenizer = RewardModel.load_tokenizer(config.reward)
            a_tokenizer = ActorModel.load_tokenizer(config.actor)
            c_tokenizer = CriticModel.load_tokenizer(config.critic)
            # safety tokens
            safety_tokens = config.actor.additonal_prompt_tokens

        elif isinstance(config, ConfigActor):
            print("Start cleaning the dataset for Actor")
            # constraint
            a_model_max_seq_len = config.max_sequence_length
            # dataset
            dataset_path = config.train_dataset_path
            # tokenizer
            a_tokenizer = ActorModel.load_tokenizer(config)
            # safety tokens
            safety_tokens = config.additonal_prompt_tokens

        elif isinstance(config, ConfigReward):
            print("Start cleaning the dataset for Reward")
            # constraint
            r_model_max_seq_len = config.max_sequence_length
            # dataset
            dataset_path = config.train_dataset_path
            # tokenizer
            r_tokenizer = RewardModel.load_tokenizer(config)

        # if there is the datasets
        if os.path.exists(dataset_path):

            # load the dataset
            with open(dataset_path, "r") as f:
                conversations = json.load(f)

            # sort in desceding order - longest first
            if isinstance(config, Config):
                conversations = BaseDataset.sort_conversation(
                    conversations,
                    only_input=True,
                    reverse=True,
                )
            else:
                conversations = BaseDataset.sort_conversation(
                    conversations,
                    only_input=False,
                    reverse=True,
                )

            old_len = len(conversations)
            # remove too long examples
            # since datasets are ordered by the length
            # we can remove the first elements until we find
            # an example that is not too long
            while len(conversations) > 0:

                # get the text to be tokenized
                if isinstance(config, Config):
                    text = conversations[0]["user_input"]
                else:
                    text = (
                        conversations[0]["user_input"]
                        + conversations[0]["completion"]
                    )

                # remove elements from RLHF dataset
                if isinstance(config, Config):
                    a_tokens = a_tokenizer.encode(text, truncation=False)
                    r_tokens = r_tokenizer.encode(text, truncation=False)
                    c_tokens = c_tokenizer.encode(text, truncation=False)
                    if (
                        len(a_tokens) + min_completion + safety_tokens
                        > a_model_max_seq_len
                    ):
                        conversations.pop(0)
                    elif (
                        len(r_tokens) + min_completion + safety_tokens
                        > r_model_max_seq_len
                    ):
                        conversations.pop(0)
                    elif (
                        len(c_tokens) + min_completion + safety_tokens
                        > c_model_max_seq_len
                    ):
                        conversations.pop(0)
                    else:
                        break

                # remove elements from Actor dataset
                elif isinstance(config, ConfigActor):
                    tokens = a_tokenizer.encode(text, truncation=False)
                    if len(tokens) + safety_tokens > a_model_max_seq_len:
                        conversations.pop(0)
                    else:
                        break

                # remove elements from Reward dataset
                elif isinstance(config, ConfigReward):
                    tokens = r_tokenizer.encode(text, truncation=False)
                    if len(tokens) > r_model_max_seq_len:
                        conversations.pop(0)
                    else:
                        break

            # if the number of examples has changed
            if len(conversations) != old_len:
                print("Number of examples before cleaning: ", old_len)
                print(
                    "Number of examples after cleaning: ", len(conversations)
                )

                # remove the old dataset
                os.remove(dataset_path)

                # save the new dataset
                with open(dataset_path, "w") as f:
                    json.dump(conversations, f, indent=4)
            else:
                print("Dataset is already clean")

        else:
            print(
                f"Dataset not found at {dataset_path}"
                f" Skipping cleaning of the dataset"
            )


class StanfordNLPSHPDataset(BaseDataset):
    """Class for Stanford NLP SHP dataset from HuggingFace"""

    def __init__(
        self,
    ) -> None:
        print("Download the dataset")
        self.dataset = load_dataset("stanfordnlp/SHP")
        print("Download Completed")

    def reformat_dataset(self, data: List) -> List[Dict]:
        """Reformat the dataset to the format required by RLHF

        Args:
            data (List): dataset from HuggingFace

        Returns:
            List[Dict]: reformatted dataset
        """

        # initialize conversations
        conversations = []

        # loop over the dataset
        for i, d in enumerate(data):
            if d["score_A"] > d["score_B"]:
                response = d["human_ref_A"]
            else:
                response = d["human_ref_B"]

            # compose user_input template
            user_input = d["history"].rstrip("\n")
            user_input = "Human: " + d["history"] + "\n\n##\n\n"

            # compose completion template
            completion = "Assistant: " + response
            conv = {
                "user_input": user_input,
                "completion": completion,
                "score": None,
            }
            conversations.append(conv)

        return conversations

    def save_dataset(
        self, dataset_folder: str, number_of_samples: int, reverse: bool = True
    ) -> None:
        """Save the dataset in the format required by RLHF

        Args:
            dataset_folder (str): path to the folder where the dataset
                will be saved
            number_of_samples (int): number of samples to take from the
                dataset
            reverse (bool, optional): sort the dataset in descending order.
                Defaults to True.
        """

        print("Generate datasets for RLHF")

        # take the train and test dataset to create the finetuning dataset
        conversations = self.reformat_dataset(self.dataset["train"])
        conversations.extend(self.reformat_dataset(self.dataset["test"]))

        # sort conversations by length of user_input + completion
        conversations = self.sort_conversation(conversations, reverse=reverse)

        # save actor training data
        with open(f"{dataset_folder}/actor_training_data.json", "w") as f:
            json.dump(conversations, f, indent=4)

        # take N samples and sort them
        conversations = self.take_n_samples(conversations, number_of_samples)
        conversations = self.sort_conversation(conversations, reverse=reverse)

        # save reward training data
        with open(f"{dataset_folder}/reward_training_data.json", "w") as f:
            json.dump(conversations, f, indent=4)

        # take the validation dataset for rlhf
        conversations = self.reformat_dataset(self.dataset["validation"])
        # sort the validation dataset
        conversations = self.sort_conversation(
            conversations,
            only_input=True,
            reverse=reverse,
        )
        # save rlhf training data
        with open(f"{dataset_folder}/rlhf_training_data.json", "w") as f:
            json.dump(conversations, f, indent=4)

        print("Generation Completed")


class AnthropicRLHF(BaseDataset):
    def __init__(
        self,
    ) -> None:

        print("Download the dataset")
        self.dataset = load_dataset("Anthropic/hh-rlhf")
        print("Download Completed")

    def reformat_dataset(self, data: List) -> List[Dict]:
        """Reformat the dataset to the format required by RLHF

        Args:
            data (List): dataset from HuggingFace

        Returns:
            List[Dict]: reformatted dataset
        """

        conversations = []
        for _, d in enumerate(data):
            current_conv = d["chosen"]
            split_answer = current_conv.split("Assistant:")

            # take all the list element in split_answer except the last one
            # and joing them with "Assistant:" in a unique string
            previous_convers = split_answer[0]
            for i, s in enumerate(split_answer[1:-1]):
                previous_convers += "Assistant:" + s

            # remove the last characters if they are "\n" from the previous
            # conversation
            previous_convers = previous_convers.rstrip("\n")
            user_input = previous_convers + "\n\n##\n\n"
            completion = "Assistant: " + split_answer[-1]

            conv = {
                "user_input": user_input,
                "completion": completion,
                "score": None,
            }

            conversations.append(conv)
        return conversations

    def save_dataset(
        self, dataset_folder: str, number_of_samples: int, reverse: bool = True
    ) -> None:
        """Save the dataset in the format required by RLHF

        Args:
            dataset_folder (str): path to the folder where the dataset
                will be saved
            number_of_samples (int): number of samples to take from the
                dataset
            reverse (bool, optional): sort the dataset in descending order.
                Defaults to True.
        """

        print("Generate datasets for RLHF")

        # generate actor and reward dataset
        conversations = self.reformat_dataset(self.dataset["train"])
        conversations = self.sort_conversation(conversations, reverse=reverse)

        # save actor training data
        with open(f"{dataset_folder}/actor_training_data.json", "w") as f:
            json.dump(conversations, f, indent=4)

        # sample N number of index from 0 to len(conversations)
        conversations = self.take_n_samples(conversations, number_of_samples)
        conversations = self.sort_conversation(conversations, reverse=reverse)

        # save reward training data
        with open(f"{dataset_folder}/reward_training_data.json", "w") as f:
            json.dump(conversations, f, indent=4)

        # rlhf dataset
        conversations = self.reformat_dataset(self.dataset["test"])

        # sort conversations by length of user_input
        conversations = self.sort_conversation(
            conversations, only_input=True, reverse=reverse
        )

        # save rlhf training data
        with open(f"{dataset_folder}/rlhf_training_data.json", "w") as f:
            json.dump(conversations, f, indent=4)

        print("Generation Completed")


class StanfordNLPSHPRewardDataset(BaseDataset):
    """Class for Stanford NLP SHP dataset from HuggingFace"""

    def __init__(
        self,
    ) -> None:
        print("Download the dataset")
        self.dataset = load_dataset("stanfordnlp/SHP")
        print("Download Completed")

    def reformat_dataset(self, data: List) -> List[Dict]:
        """Reformat the dataset to the format required by RLHF

        Args:
            data (List): dataset from HuggingFace

        Returns:
            List[Dict]: reformatted dataset
        """
        
        def get_score_winning_answer(x):

            return int(
                (
                    min(x["score"], upper_whisker[x["post_id"]])
                    / min(max_vote[x["post_id"]], upper_whisker[x["post_id"]])
                ) * 5
            )

        data = data.to_pandas()

        A_answers = data[["c_root_id_A", "score_A", "post_id", "history", "human_ref_A"]]
        B_answers = data[["c_root_id_B", "score_B", "post_id", "history", "human_ref_B"]]

        # Take both answers A and B
        A_answers.rename(
            columns={
                "c_root_id_A": "c_id",
                "score_A": "score",
                "history": "user_input",
                "human_ref_A": "completion",
            },
            inplace=True,
        )
        B_answers.rename(
            columns={
                "c_root_id_B": "c_id",
                "score_B": "score",
                "history": "user_input",
                "human_ref_B": "completion",
            },
            inplace=True,
        )
        conversations = pd.concat([A_answers, B_answers], axis=0)

        # Removing duplicates so that each answer is used only once
        conversations.drop_duplicates(subset=["c_id"], inplace=True)

        # Computing for each post the upper whisker as Q3 + 1.5 * IQR
        upper_whisker = conversations.groupby(by=["post_id"]).agg(
            {"score": lambda x: x.quantile(0.75) + 1.5 * (x.quantile(0.75) - x.quantile(0.25))}
        )["score"]
        
        max_vote = conversations.groupby(by=["post_id"]).agg({"score": max})["score"]
        
        norm_score = conversations.apply(get_score_winning_answer, axis=1)

        conversations["reward"] = norm_score

        conversations = conversations[["user_input", "completion", "reward"]].rename({"reward":"score"}).to_dict("records")

        return conversations

    def save_dataset(
        self, dataset_folder: str, number_of_samples: int, reverse: bool = True
    ) -> None:
        """Save the dataset in the format required by RLHF

        Args:
            dataset_folder (str): path to the folder where the dataset
                will be saved
            number_of_samples (int): number of samples to take from the
                dataset
            reverse (bool, optional): sort the dataset in descending order.
                Defaults to True.
        """

        print("Generate reward datasets")

        # take the train and test dataset to create the finetuning dataset
        conversations = self.reformat_dataset(self.dataset["train"])
        conversations.extend(self.reformat_dataset(self.dataset["test"]))

        # sort conversations by length of user_input + completion
        conversations = self.sort_conversation(conversations, reverse=reverse)

        # take N samples and sort them
        conversations = self.take_n_samples(conversations, number_of_samples)
        conversations = self.sort_conversation(conversations, reverse=reverse)

        # save reward training data
        with open(f"{dataset_folder}/reward_training_data.json", "w") as f:
            json.dump(conversations, f, indent=4)

        print("Generation Completed")