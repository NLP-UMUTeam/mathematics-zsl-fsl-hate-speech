import re
import pandas as pd 
import argparse
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline, AutoModelForCausalLM, BitsAndBytesConfig
import gc
from sentence_transformers import InputExample, SentenceTransformer, losses, util


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

mistral_prompt = """\
<s>[INST]Here are a couple of examples of labels assigned by experts: {examples}. Please respond with yes or no. Is the following tweet written in english sexist speech or not? '{text}' [/INST]\
"""

zephyr_prompt = """\
<|system|>\nYou are a classification model that is really good at following instructions and produces brief answers that users can use as data right away. Here are a couple of examples of labels assigned by experts: {examples}.
Please respond only with yes or no.</s>\n<|user|>\n Is the following tweet written in english sexist speech or not? '{text}'</s>\n<|assistant|>\
"""

tulu_prompt = """\
<|user|>\nHere are a couple of examples of labels assigned by experts: {examples}. Please respond only with yes or no. Is the following tweet written in english sexist speech or not? '{text}'</s>\n<|assistant|>\n\
"""


llama_prompt = """
### System:
You are a classification model that is really good at following
instructions and produces brief answers that users can use as data right away.
Please respond only with yes or no.
Here are a couple of examples of labels assigned by experts:
{examples}
### User:
Is the following tweet written in english sexist speech or not?
'{text}'
### Assistant:
"""

example_prompt = """
Examples of texts labelled '{label}':
{examples}
"""
 
    
bnb_config = BitsAndBytesConfig(
    load_in_8bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

def clean_gpu_cache(): 
    print("Se ha liberado la memoria")
    torch.cuda.empty_cache()
    gc.collect()
    
    
class ZeroShotClassifier(): 

    def __init__(self, model_name, prompt, device, model_type, examples): 
        if model_type not in {'llama-2', 'zephyr', 'mistral', 'tulu'}:
            raise Exception('Invalid model')
        self.model_name = model_name 
        self.prompt = prompt
        self.device = device
        self.model_type = model_type 
        self.examples_prompt = get_examples_prompt(examples)
        
        self.tokenizer= AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name, quantization_config=bnb_config, use_cache=False, device_map="auto")
        self.tokenizer.pad_token = self.tokenizer.eos_token

        
    def run_prompt(self, prompt, text, classes): 
        inputs = self.tokenizer(self.prompt.format(examples=self.examples_prompt, text=text), return_tensors="pt").to("cuda")
        output = self.model.generate(**inputs, do_sample=True, top_p=0.95, top_k=0, max_new_tokens=256, pad_token_id=self.tokenizer.eos_token_id)
        output = self.tokenizer.decode(output[0], skip_special_tokens=True)

        # output = self.tokenizer.decode(self.model.generate(input_ids)[0], do_sample=True, temperature=0.1, top_p=0.95, top_k=0, max_new_tokens=1026)
        return output 
            
            
    def predict (self, text, classes): 
        output = self.run_prompt(self.prompt, text, classes)
        if self.model_type == 'llama-2': 
            output = output.split("### Assistant:")[1].strip().lower()
            print(output)
        elif self.model_type == 'zephyr':
            output = output.split("<|assistant|>")[1].strip().lower()
            print(output)
        elif self.model_type == 'mistral':
            output = output.split("[/INST]")[1].strip().lower()
            print(output)
        elif self.model_type == 'tulu':
            output = output.split("<|assistant|>")[1].strip().lower()
            print(output)
    
    
    def predict_list(self, text_list, classes, few_type, num_few): 
        if self.model_type == 'llama-2': 
            prediction = []
            for text in tqdm(text_list):
                val_df = pd.DataFrame()
                output = self.run_prompt(self.prompt, text, classes)
                output = output.split("### Assistant:")[1].strip().lower()
                if  re.search(r'(?<!\S)yes\b', output.lower()):
                    prediction.append(classes[0])
                elif re.search(r'(?<!\S)no\b', output.lower()):
                    prediction.append(classes[1])
                else: 
                    prediction.append("")
                    print("No founded", output)
                    
                # Iterar save
                val_df['prediction'] = prediction 
                val_df.to_csv(f"few-shot-results/{self.model_type}_few_{few_type}_{num_few}_prediction.csv", index=False)
            return prediction 
            
        elif self.model_type == 'zephyr' or self.model_type == 'tulu': 
            prediction = []
            for text in tqdm(text_list):
                val_df = pd.DataFrame()
                output = self.run_prompt(self.prompt, text, classes)
                output = output.split("<|assistant|>")[1].strip().lower()
                if  re.search(r'(?<!\S)yes\b', output.lower()):
                    prediction.append(classes[0])
                elif re.search(r'(?<!\S)no\b', output.lower()):
                    prediction.append(classes[1])
                else: 
                    prediction.append("")
                    print("No founded", output)
                    
                val_df['prediction'] = prediction 
                val_df.to_csv(f"few-shot-results/{self.model_type}_few_{few_type}_{num_few}_prediction.csv", index=False)
            return prediction 
            
        elif self.model_type == 'mistral': 
            prediction = []
            for text in tqdm(text_list):
                val_df = pd.DataFrame()
                output = self.run_prompt(self.prompt, text, classes)
                output = output.split("[/INST]")[1].strip().lower()
                if  re.search(r'(?<!\S)yes\b', output.lower()):
                    prediction.append(classes[0])
                elif re.search(r'(?<!\S)no\b', output.lower()):
                    prediction.append(classes[1])
                else: 
                    prediction.append("")
                    print("No founded", output)
                
                val_df['prediction'] = prediction 
                val_df.to_csv(f"few-shot-results/{self.model_type}_few_{few_type}_{num_few}_prediction.csv", index=False)
                
            return prediction 


def get_examples_prompt (examples):
    text_examples = []
    for label, examples in examples.items():
        examples = [f"'{example}'" for example in examples]
        subprompt = example_prompt.format(
            label=label, examples="\n".join(examples)
        )
        text_examples.append(subprompt)
    examples_subprompt = "\n".join(text_examples)
    return examples_subprompt
    


def get_few_shot_predictions(model_name, val_df, text_list, classes, examples, prompt, model_type, few_type, num_few, save_path):
    zero_classifier = ZeroShotClassifier(model_name, prompt, device, model_type, examples)
    # predictions = zero_classifier.predict(text_list[0], classes)
    predictions = zero_classifier.predict_list(text_list, classes, few_type, num_few)
    # val_df['prediction'] = predictions 
    # val_df.to_csv(f"{save_path}/{model_type}_few_{few_type}_{num_few}_prediction.csv", index=False)
    

def get_random_examples (train_df, num_few, classes, column_name): 
    val_df_true = train_df[train_df['label'] == classes[0]].head(num_few)
    val_df_false = train_df[train_df['label'] == classes[1]].head(num_few)
    
    examples = {classes[0]: val_df_true[column_name].tolist(), classes[1]: val_df_false[column_name].tolist()}
    return examples
    
    
def get_sentences_by_st(dataset, positive_question, negative_question, classes):
    model = SentenceTransformer("hiiamsid/sentence_similarity_spanish_es")
    positive_df = {}
    negative_df = {}
    
    for tweet, label in zip(dataset['tweet'].tolist(), dataset['label'].tolist()):
        if "tweet" not in positive_df: 
            positive_df["tweet"] = []
        if "score" not in positive_df:
            positive_df["score"] = []
        
        if "tweet" not in negative_df: 
            negative_df["tweet"] = []
        if "score" not in negative_df:
            negative_df["score"] = []
        
        if label == classes[0]:
            score = util.cos_sim(model.encode(positive_question), model.encode(tweet))[0].tolist()
            positive_df["tweet"].append(tweet)
            positive_df["score"].append(score)
            
        elif label == classes[1]:
            score = util.cos_sim(model.encode(negative_question), model.encode(tweet))[0].tolist()            
            negative_df["tweet"].append(tweet)
            negative_df["score"].append(score)
            
    positive_df = pd.DataFrame.from_dict(positive_df).sort_values(by=['score'], ascending=False)    
    negative_df = pd.DataFrame.from_dict(negative_df).sort_values(by=['score'], ascending=False)  
    print(positive_df)
    print(negative_df)
    return positive_df, negative_df
    
    
def get_st_examples (train_df, num_few, positive_question, negative_question, classes, column_name): 
    positive_df, negative_df = get_sentences_by_st(train_df, positive_question, negative_question, classes)
    val_df_true = positive_df.head(num_few)
    val_df_false = negative_df.head(num_few)
    examples = {classes[0]: val_df_true[column_name].tolist(), classes[1]: val_df_false[column_name].tolist()}
    return examples
    
    
def main(args): 
    dataset_path = args.dataset_path
    save_path = args.save_path
    few_type = args.type
    num_few = args.num_few
    
    transformers_models = {  
                           "llama-2": "stabilityai/StableBeluga-7B",
                           "mistral": "mistralai/Mistral-7B-Instruct-v0.1", 
                           "tulu": "allenai/tulu-2-dpo-7b",
                           "zephyr": "HuggingFaceH4/zephyr-7b-beta",
                          }
    
    classes = ['sexist', 'not sexist'] 
    column_name = 'tweet'
        
    dataset = pd.read_csv(dataset_path)
    dataset = dataset[['tweet', '__split', 'label']]    
    print(dataset)
   
    train_df = dataset[dataset['__split']=='train']
    val_df = dataset[dataset['__split']=='test']
    print(val_df)
    
    if few_type == 'random':
        examples = get_random_examples(train_df, num_few, classes, column_name)
    elif few_type == 'sentences-transformers': 
        positive_question = "This is a sexist speech" 
        negative_question = "This is a feminist speech"
        examples = get_st_examples(train_df, num_few, positive_question, negative_question, classes, column_name)  
        
        
    for k, v in transformers_models.items(): 
        if k == "llama-2": 
            get_few_shot_predictions(v, val_df, val_df[column_name].tolist(), classes, examples, llama_prompt, k, few_type, num_few, save_path)
        elif k == "zephyr": 
            get_few_shot_predictions(v, val_df, val_df[column_name].tolist(), classes, examples, zephyr_prompt, k, few_type, num_few, save_path)
        elif k == "mistral": 
            get_few_shot_predictions(v, val_df, val_df[column_name].tolist(), classes, examples, mistral_prompt, k, few_type, num_few, save_path)
        elif k == "tulu": 
            get_few_shot_predictions(v, val_df, val_df[column_name].tolist(), classes, examples, tulu_prompt, k, few_type, num_few, save_path)
        clean_gpu_cache()
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', type=str, default='./edos.csv')
    parser.add_argument('--save_path', type=str, default='./zero-shot-results')
    parser.add_argument('--type', type=str, default='random')
    parser.add_argument('--num_few', type=int, default=5)
    args = parser.parse_args()
    main(args)
    