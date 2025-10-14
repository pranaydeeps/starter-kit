from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import transformers
import torch
import pandas as pd
import re
from peft import prepare_model_for_kbit_training
from transformers import AutoTokenizer, AutoModelForCausalLM
import datasets
import bitsandbytes as bnb
import huggingface_hub
from transformers.pipelines.pt_utils import KeyDataset

def Set_Prompt_Template(dataframe, prompt,tokenizer):
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
        input_text = prompt.replace("{PLACEHOLDER_FOR_INPUTTEXT}" , row['text'])

        # get text between \begin[user] and \end[user], removing these placeholder tokens
        user_text = re.search(r'\\begin\[user\](.*?)\\end\[user\]', input_text, re.DOTALL)

        # create a list with interactions between user and assistant
        # for zero-shot this is only the input
        chat = []
        chat.append({"role": "user","content": user_text.group(0).replace(r"\begin[user]", "").replace(r"\end[user]", "").replace("  "," ").capitalize()})
        
        # this function automatically sets up the chat tokens for instruction tuning based on the model tokenizer.
        input_chat = tokenizer.apply_chat_template(chat, tokenize=False)
        full_explanations.append(input_chat)
    #add to dataframe
    dataframe["input"] = full_explanations
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
    base_model.config.use_cache = False
    base_model = prepare_model_for_kbit_training(base_model)
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=access_token)
    tokenizer.pad_token = tokenizer.eos_token
    # for Llama models the padding side needs to be set to right, check for other models
    tokenizer.padding_side = "right"

    # TODO: replace this prompt for your task, this will be asked during inference for each sample
    prompt = r"\begin[user]Explain why this tweet is ironic:\n\n### Text: {PLACEHOLDER_FOR_INPUTTEXT}\end[user]"
    #train_data = pd.read_csv(f'data/{language}/train.csv')

    # load in the test data
    test_data = pd.read_csv(f'data/test.csv')
    test_data["labels"] = test_data["explanation"] #.apply(lambda x: x.lower() if type(x) == str else x)

    # set up the prompt template
    test_data = Set_Prompt_Template(dataframe=test_data, prompt=prompt, tokenizer=tokenizer)
    testset = datasets.Dataset.from_pandas(test_data)

    print("Input example")
    print(test_data["input"].to_list()[0])
    generated_outputs = []
    pipe = transformers.pipeline(
        "text-generation",
        model=base_model,
        tokenizer=tokenizer,
        torch_dtype=torch.float16,
        device_map="auto",do_sample=True, top_k=10, num_return_sequences=1, eos_token_id=tokenizer.eos_token_id, max_new_tokens=150)
    
    for out in pipe(KeyDataset(testset, "input"), batch_size=1):
        print("Output:")
        print(out)
        generated_outputs.append(out)
    
    print("Inference complete, saving results")
    test_data["generated_output"] = generated_outputs

    test_data.to_csv(f"data/generative_test_output_{save_as_name}_zeroshot.csv", index=False)
    print(f"Saved results to data/generative_test_output_{save_as_name}_zeroshot.csv")

if __name__ == "__main__":
    main()
    print("Complete")