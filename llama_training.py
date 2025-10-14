from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, TrainingArguments
import transformers
import torch
import pandas as pd
import re
from peft import LoraConfig, prepare_model_for_kbit_training
from peft import prepare_model_for_kbit_training
from transformers import AutoTokenizer, AutoModelForCausalLM
import datasets
import bitsandbytes as bnb
import huggingface_hub
from transformers.pipelines.pt_utils import KeyDataset
from trl import SFTTrainer


def find_all_linear_names(model): # copied from https://github.com/mzbac/llama2-fine-tune/blob/master/utils.py
    cls = bnb.nn.Linear4bit
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    return list(lora_module_names)

def print_trainable_parameters(model): # copied from https://github.com/mzbac/llama2-fine-tune/blob/master/utils.py
  """
  Prints the number of trainable parameters in the model.
  """
  trainable_params = 0
  all_param = 0
  for _, param in model.named_parameters():
    all_param += param.numel()
    if param.requires_grad:
      trainable_params += param.numel()
  print(
      f"trainable params: {trainable_params} || all params: {all_param} || trainables%: {100 * trainable_params / all_param}"
  )

def Set_Prompt_Template(dataframe, prompt, tokenizer, train=True):
    """ 
    This function prepares the prompt template for inference. You provide the full dataframe and th"""
    full_explanations = []
    # get an example for each unique label
    for i, row in dataframe.iterrows():
        # TODO: set up the system prompt, this can include personality description AND general guidelines.
        systemprompt =" You are an expert trained in identifying irony and sarcasm in social media text and explaining the underlying reasoning."
        # adding general task description.
        systemprompt += " Your task is to explain why tweets should be considered ironic.\n\
            Make sure to base your explanation on background knowledge that is not present in the text itself.\n\
            This background knowledge can include common assumptions, factual knowledge and social conventions.\n"

        # get the actual to classify text and insert it into the prompt
        sample_inprompt = prompt.replace("{PLACEHOLDER_FOR_INPUTTEXT}" , row['text']).replace("{PLACEHOLDER_FOR_LABEL}", row['task_labels'])

        # get text between \begin[user] and \end[user], removing these placeholder tokens
        user_text = re.search(r'\\begin\[user\](.*?)\\end\[user\]', sample_inprompt, re.DOTALL)

        # get text for system , this part is ignored for testing/inference only used as gold for training
        system_text = re.search(r'\\begin\[assistant\](.*?)\\end\[assistant\]', sample_inprompt, re.DOTALL)
        

        # create a list with interactions between user and assistant
        # for zero-shot this is only the input
        chat = []
        chat.append({"role": "user","content": user_text.group(0).replace(r"\begin[user]", "").replace(r"\end[user]", "").replace("  "," ").capitalize()})

        # provide the label for the model
        if train:
            chat.append({"role": "assistant","content": system_text.group(0).replace(r"\begin[assistant]", "").replace(r"\end[assistant]", "").replace("  "," ").capitalize()})
        
        # this function automatically sets up the chat tokens for instruction tuning based on the model tokenizer.
        input_chat = tokenizer.apply_chat_template(chat, tokenize=False)
        full_explanations.append(input_chat)

    #add to dataframe, have to add to "text" column for training
    dataframe["text"] = full_explanations
    return dataframe

def main():
    # TODO: set your huggingface token here
    access_token = "hf_placeholder"
    huggingface_hub.login(token=access_token)

    # TODO: set your model here
    # take care, you need transformers version 4.43, 4.45 is not yet supported, same for ipex-llm==2.1.0b2
    model_name = "meta-llama/Llama-3.2-3B-Instruct"
    # removes the repository name (here meta-llama) for saving the output file
    save_as_name = model_name.split("/")[1]

    #set up quantization config
    bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=False,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )
    # load model and tokenizer with quantization
    base_model = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=bnb_config, token=access_token)
    #base_model.config.use_cache = False
    base_model = prepare_model_for_kbit_training(base_model)
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=access_token)
    tokenizer.pad_token = tokenizer.eos_token
    # for Llama models the padding side needs to be set to right, check for other models
    tokenizer.padding_side = "right"

    # TODO: replace this prompt for your task, this will be asked during inference for each sample
    # NOTE: here we also have a placeholder for the label, this is only used during training
    prompt = r"\begin[user]Explain why this tweet is ironic:\n\n### Text: {PLACEHOLDER_FOR_INPUTTEXT}\end[user]\begin[assistant]### Explanation:{PLACEHOLDER_FOR_LABEL}\end[assistant]"
    #train_data = pd.read_csv(f'data/{language}/train.csv')

    # load in the test data
    train_data = pd.read_csv(f'data/test.csv')
    # NOTE: the labels are text outputs here, if your labels are numbers, you should convert them to text first
    train_data["task_labels"] = train_data["explanation"] #.apply(lambda x: x.lower() if type(x) == str else x)

    # set up the prompt template
    train_data = Set_Prompt_Template(dataframe=train_data, prompt=prompt, tokenizer=tokenizer)
    trainset = datasets.Dataset.from_pandas(train_data)

    # Set up PEFT LoRA for fine-tuning.
    lora_config = LoraConfig(
        lora_alpha=16,
        r=32,
        target_modules=find_all_linear_names(base_model),
        task_type="CAUSAL_LM",
    )
    #max_seq_length = 1024

    trainer = SFTTrainer(
        model=base_model,
        train_dataset=trainset,
        args=TrainingArguments(
            per_device_train_batch_size=1,  # This is actually the global batch size for SPMD.
            num_train_epochs=2,
            output_dir=f"./trained_{save_as_name}",
            eval_accumulation_steps=10,
            dataloader_drop_last = True,  # Required for SPMD.
            hub_private_repo=True,
        ),
        peft_config=lora_config
    )
    
    trainer.train()
    # create folder if it does not exist
    import os
    if not os.path.exists(f"finetuned_models"):
        os.makedirs(f"finetuned_models")
    # saves the model locally and pushes to the hub
    trainer.save_model(f"finetuned_models/{save_as_name}")
    trainer.push_to_hub(token=access_token)
    # also saves the tokenizer on the same path
    #tokenizer.save_pretrained(f"finetuned_models/{save_as_name}")
    tokenizer.push_to_hub(f"Amala3/{save_as_name}_test", token=access_token, public=False)

    # after this, you can use the inference script while loading in your finetuned model



if __name__ == "__main__":
    main()
    print("Complete")