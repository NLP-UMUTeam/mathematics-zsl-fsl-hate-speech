import re
import pandas as pd 
import argparse
import torch
from tqdm import tqdm
import swifter
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline, AutoModelForCausalLM, BitsAndBytesConfig


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


generative_prompt = """
### Instruction:\nYour task will be to classify a text document into one of the following classes: {classes}.\n\n### Input:\n{text}\n\n### Response:\n
"""


llama_prompt = """
### System:\nYou are a classification model that is really good at following instructions and produces brief answers that users can use as data right away. Please follow the user's instructions as precisely as you can.\n\n### User: Your task will be to classify a text document into one of the following classes: {classes}. Please respond with a single label that you think fits the document best. Classify the following piece of text: '{text}'\n\n### Assistant:\n
"""


text2text_prompt = """
Please classify it as one of these classes: {classes}. 
Please only respond with the class label in the same format as provided.
Label this piece of text:
'{text}'
"""


mt0_prompt = """
{text}. Is this {classes} tweet? 
"""



zephyr_prompt = """
<|system|>\nYour task will be to classify a text document into one of the following classes: {classes}. Please respond with a single label that you think fits the document best.</s>\n<|user|>\nClassify the following piece of text: {text}</s>\n<|assistant|>
"""
    
    
bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    
    )
    

class ZeroShotClassifier(): 

    def __init__(self, model_name, prompt, device, model_type): 
        if model_type not in {'flan-t5', 'flan-alpaca', 'llama-2', 'mt0', 'mt5', 'flan-ul2'}:
            raise Exception('Invalid model')
        self.model_name = model_name 
        self.prompt = prompt
        self.device = device
        self.model_type = model_type 
        
        if self.model_type == 'llama-2':
            self.tokenizer= AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForCausalLM.from_pretrained(self.model_name, quantization_config=bnb_config, device_map="auto")
        
        elif self.model_type == 'flan-ul2':
            self.tokenizer= AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name, quantization_config=bnb_config, device_map="auto")
            
        else:
            self.tokenizer= AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name, device_map="auto")
        
        
    def run_prompt(self, prompt, text, classes): 
        if self.model_type == 'llama-2':
            input_ids = self.tokenizer(self.prompt.format(classes=", ".join(classes_in_quotes), text=text), return_tensors="pt", truncation=True).input_ids.cuda()
            output = self.tokenizer.decode(self.model.generate(input_ids)[0], do_sample=True, temperature=0.5, top_p=0.95, top_k=0.4, num_beams=1, max_new_tokens=256)
            return output 
            
        elif self.model_type == 'mt0':
            input_ids = self.tokenizer(self.prompt.format(classes=classes[0], text=text), return_tensors="pt").input_ids.cuda()
            output = self.tokenizer.decode(self.model.generate(input_ids)[0], skip_special_tokens=True)
            return output   
            
        else:
            input_ids = self.tokenizer(self.prompt.format(classes=", ".join(classes), text=text), return_tensors="pt").input_ids.cuda()
            output = self.tokenizer.decode(self.model.generate(input_ids)[0], skip_special_tokens=True)
            return output
            
            
    def predict (self, text, classes): 
        output = self.run_prompt(self.prompt, text, classes)
        if self.model_type == 'llama-2': 
            print(output)
            output = output.split("### Assistant:")[1].strip().lower()
            for idx, c in enumerate(classes): 
                    if re.search(fr'(?<!\S){c}\b', output): 
                        return c
                    elif idx == len(classes) - 1: 
                        return ""
        
        elif self.model_type == 'mt0': 
            print(output)
        else:
            print(output)
    
    
    def predict_list(self, text_list, classes): 
        if self.model_type == 'llama-2': 
            prediction = []
            for text in tqdm(text_list):
                output = self.run_prompt(self.prompt, text, classes)
                print(output)
                output = output.split("### Assistant:")[1].strip().lower()
                for idx, c in enumerate(classes): 
                    if re.search(fr'(?<!\S){c}\b', output): 
                        prediction.append(c)
                        break
                    elif idx == len(classes) - 1: 
                        prediction.append("") 
                        print("No founded", output)
            return prediction 
            
        elif self.model_type == 'mt0': 
            prediction = []
            for text in tqdm(text_list):
                output = self.run_prompt(self.prompt, text, classes)   
                if  re.search(r'(?<!\S)yes\b', output.lower()):
                    prediction.append(classes[0])
                elif re.search(r'(?<!\S)no\b', output.lower()):
                    prediction.append(classes[1])
                else: 
                    prediction.append("")
                    print("No founded", output)
            return prediction 
            
        else:
            prediction = []
            for text in tqdm(text_list):
                output = self.run_prompt(self.prompt, text, classes)
                output = output.lower()
                for idx, c in enumerate(classes): 
                    if re.search(fr'(?<!\S){c}\b', output): 
                        print("salida", output)
                        print(c)
                        prediction.append(c)
                        break
                    elif idx == len(classes) - 1: 
                        prediction.append("")
                        print("No founded", output)
            return prediction 
    


def get_zero_shot_predictions(model_name, val_df, text_list, classes, prompt, model_type, dataset_name, save_path):
    zero_classifier = ZeroShotClassifier(model_name, prompt, device, model_type)
    predictions = zero_classifier.predict_list(text_list, classes)
    val_df['prediction'] = predictions 
    val_df.to_csv(f"{save_path}/{model_type}_{dataset_name}_val_prediction.csv", index=False)
    

def main(args): 
    dataset_path = args.dataset_path
    save_path = args.save_path
    
    transformers_models = {"flan-alpaca": "declare-lab/flan-alpaca-gpt4-xl",
                           "flan-t5": "google/flan-t5-xl",
                           "mt0": "bigscience/mt0-large" 
                          }
    
    # model_name = "stabilityai/StableBeluga-7B"
    # model_name = "google/flan-ul2"
    
    # Exist
    # classes = ['sexist', 'non-sexist'] 
    # Hateval
    classes = ['hate', 'non-hate'] 
    # misocorpus 
    # classes = ['misogyny', 'non-misogyny']
    # haternet and hasoc
    # classes = ['hateful', 'non-hateful']
    column_name = 'tweet'
        
    dataset = pd.read_csv(dataset_path)
    # For hasoc
    # val_df = dataset
    # column_name = 'text'
    dataset = dataset[['tweet', '__split', 'label']]    
   
    train_df = dataset[dataset['__split']=='train']
    val_df = dataset[dataset['__split']=='test']
    
    for k, v in transformers_models.items(): 
        if k == "flan-alpaca": 
            get_zero_shot_predictions(v, val_df, val_df[column_name].tolist(), classes, text2text_prompt, k, "hateval-en", save_path)
            
        elif k == "flan-t5": 
            get_zero_shot_predictions(v, val_df, val_df[column_name].tolist(), classes, text2text_prompt, k, "hateval-en", save_path)
            
        elif k == "mt0": 
            get_zero_shot_predictions(v, val_df, val_df[column_name].tolist(), classes, mt0_prompt, k, "hateval-en", save_path)
            
          
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', type=str, default='./hateval-en.csv')
    parser.add_argument('--save_path', type=str, default='./results_hateval-en')
    args = parser.parse_args()
    main(args)
    