import argparse
import json

from langchain import OpenAI, LLMChain, PromptTemplate, HuggingFaceHub
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

class ScoreGenerator:
    def __init__(
        self,
        llm_model: str,
        llm_temperature: float,
        llm_max_tokens: int,
        llm_local_dir: str,
        reward_template: dict,
    ) -> None:

        self.llm_max_tokens = llm_max_tokens
        self.llm_temperature = llm_temperature
        self.llm_model = llm_model
        self.llm_local_dir = llm_local_dir
        self.reward_template = reward_template

        # Customaize your own Reward template by changing the
        # prompt_template
        prompt_template = PromptTemplate(**reward_template)
        print(prompt_template)

        # initialize LLM and LangChain
        if llm_local_dir == None:
            if llm_model == "flan-t5-xl":
                hf_llm=HuggingFaceHub(repo_id="google/flan-t5-xl", model_kwargs={"temperature":llm_temperature, "max_length":llm_max_tokens, "raw_response":True})
                self.llm = LLMChain(llm=hf_llm, prompt=prompt_template)
            else:
                openai_llm = OpenAI(
                    model_name=llm_model,
                    temperature=llm_temperature,
                    max_tokens=llm_max_tokens,
                )
                self.llm = LLMChain(llm=openai_llm, prompt=prompt_template) 
        elif llm_model != "flan-t5-xl":
            raise Exception('Only local LLM supported is HuggingFace’s flan-t5-xl model, try again with -m flan-t5-xl')

    def distill(
        self,
        dataset_path: str,
    ) -> None:
        """Parse the dataset and assign scores using LLMs
        then save back the dataset with the uploaded scores
        """
        model_path = self.llm_local_dir
        model = AutoModelForSeq2SeqLM.from_pretrained(model_path,temperature=self.llm_temperature, max_length=self.llm_max_tokens)
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        print("Assigning scores to the reward dataset...")

        # load the dataset
        with open(dataset_path, "r") as f:
            train_data = json.load(f)

        # for each element of the dataset, assing a score.
        for i, data in enumerate(train_data):
            if data.get("score", None) is None:

                user_input = data["user_input"]
                completion = data["completion"]
                print(
                    f"#### Data {i}:\n"
                    f"#### User_input:\n {user_input}\n"
                    f"#### Completion:\n {completion}\n"
                )
                if self.llm_local_dir == None:
                    prompt_tokens = (
                        data["user_input"]
                        + data["completion"]
                        + self.reward_template["template"]
                        + self.llm.prompt.template
                    )
                else:
                    prompt_tokens = (data["user_input"] + data["completion"] + REWARD_TEMPLATE["template"])
                prompt_len = int(len(prompt_tokens.split(" ")) / 0.75)
                # 80% of the max length as safety margin
                if prompt_len > self.llm_max_tokens * 0.8:
                    print(
                        f"The prompt of the data {i} is too long\n"
                        f"tokens: {prompt_len}\n"
                        f"max_tokens: {self.llm_max_tokens * 0.8}"
                    )
                    continue
                
                if self.llm_local_dir == None:
                    score = self.llm.run(
                        user_input=data["user_input"],
                        completion=data["completion"],
                        raw_response=True,
                    ).strip()
                else:
                    inputs = tokenizer(prompt_tokens, padding=True, max_length=self.llm_max_tokens, truncation=True, return_tensors="pt")
                    out = tokenizer.batch_decode(model.generate(**inputs),skip_special_tokens=True)
                    score = out[0]
                # TODO: extract from score the float value with a regex
                try:
                    score = float(score)
                except Exception:
                    print(
                        f"The score returned by the LLM for the"
                        f"data, {i}, is not a float float:\n{score}"
                    )
                    continue
                data["score"] = score
                print(f"### Score: {score} \n\n")
        # remove all the data that have no score
        train_data = [data for data in train_data if data.get("score", None)]
        # save the dataset back
        print("Writing the updated dataset back to disk ... ")
        with open(dataset_path, "w") as f:
            json.dump(train_data, f)
        print("Score Assignment Completed")


if __name__ == "__main__":

    REWARD_TEMPLATE = dict(
        template=(
            "You have to evaluate the following chat with a score"
            "between 0 and 5"
            "You MUST evaluate: text quality, content quality and"
            "coherence.\n"
            "You MUST return only the number that represents your"
            "judgment.\n"
            "The input of the user is: {user_input}\n"
            "The output of the chatbot is: {completion}\n"
            "The score is:\n"
        ),
        input_variables=["user_input", "completion"],
    )

    # Setup argument parser
    parser = argparse.ArgumentParser(
        prog="generate_rewards.py",
        description="Generate rewards using LangChain and LLMs",
    )

    parser.add_argument("dataset_path", help="Path to the dataset")
    parser.add_argument(
        "-m",
        "--model",
        help="Specify the model to be used",
        default="text-davinci-003",
    )
    parser.add_argument(
        "-t",
        "--temperature",
        help="Specify the temperature of the score assignment",
        default=0.5,
    )
    parser.add_argument(
        "-k",
        "--max_tokens",
        help="Specify the max tokens of the score assignement",
        default=2048,
    )
    parser.add_argument(
        "-r",
        "--reward_template",
        help="Specify the reward template to be used",
        default=None,
    )
    parser.add_argument(
        "-l",
        "--local_directory",
        help="Specify the directory of the locally stored LLM to be used",
        default=None,
    )

    # parse arguments
    args = parser.parse_args()

    if args.reward_template:
        templates = json.loads(args.reward_template)
        if templates.get("reward", None) is None:
            rw_template = REWARD_TEMPLATE
        else:
            rw_template = templates["reward"]
    else:
        rw_template = REWARD_TEMPLATE

    score_generator = ScoreGenerator(
        args.model, args.temperature, args.max_tokens, args.local_directory, rw_template
    )

    score_generator.distill(args.dataset_path)
