import re
import pandas as pd 
import argparse
import torch
from tqdm import tqdm

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline, AutoModelForCausalLM, BitsAndBytesConfig
from stormtrooper import Text2TextFewShotClassifier, GenerativeFewShotClassifier


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


generative_prompt = """
### Instruction:\nYour task will be to classify a text document into one of the following classes: {classes}.\n\n### Input:\n{text}\n\n### Response:\n
"""


llama_prompt = """
A continuación hay una instrucción que describe una tarea. Escriba una respuesta que complete adecuadamente la solicitud.\n\n### Instrucción:\nYour task will be to classify a text document into one of the following classes: {classes}. The text is: {text}\n\n### Respuesta:\n
"""


text2text_prompt = """
I will give you a piece of text. Please classify it as one of these classes: {classes}. 
Please only respond with the class label in the same format as provided.
Here are some examples of texts labelled by experts.
{examples}
Label this piece of text:
'{text}'
"""


example_prompt = """
Examples of texts labelled '{label}':
{examples}
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
            input_ids = self.tokenizer(self.prompt.format(classes=', '.join(classes), text=text), return_tensors="pt", truncation=True).input_ids.cuda()
            output = self.tokenizer.decode(self.model.generate(input_ids)[0], do_sample=True, temperature=0.1, top_p=0.75, top_k=40, early_stopping=True, num_beams=1, max_new_tokens=150)
            return output 
            
        elif self.model_type == 'mt0':
            input_ids = self.tokenizer(self.prompt.format(classes=classes[0], text=text), return_tensors="pt").input_ids.cuda()
            output = self.tokenizer.decode(self.model.generate(input_ids)[0], skip_special_tokens=True)
            return output   
            
        else:
            classes_in_quotes = [f"'{label}'" for label in classes]
            input_ids = self.tokenizer(self.prompt.format(classes=", ".join(classes_in_quotes), text=text), return_tensors="pt").input_ids.cuda()
            output = self.tokenizer.decode(self.model.generate(input_ids)[0], skip_special_tokens=True)
            return output
            
            
    def predict (self, text, classes): 
        output = self.run_prompt(self.prompt, text, classes)
        if self.model_type == 'llama-2': 
            print(output)
            output = output.split("### Response:")[1].strip().lower()
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
                output = output.split("### Response:")[1].strip().lower()
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
                if output.lower() == 'yes': 
                    prediction.append(classes[0])
                elif output.lower() == 'no': 
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
                        print(output)
                        prediction.append(c)
                        break
                    elif idx == len(classes) - 1: 
                        prediction.append("")
                        print("No founded", output)
            return prediction 
    


def get_few_shot_prediction(model_name, val_df, text_list, examples_texts, examples_labels, dataset_name, model_type, save_path):         
    model = Text2TextFewShotClassifier(model_name, device="cuda:0")
    model.fit(examples_texts, examples_labels)
    predictions = model.predict(text_list)

    val_df['prediction'] = predictions 
    val_df.to_csv(f"{save_path}/{model_type}_{dataset_name}_few_val_prediction.csv", index=False)
    
    

def main(args): 
    dataset_path = args.dataset_path
    save_path = args.save_path
    
    trasformers_models = {'flan_alpaca': "declare-lab/flan-alpaca-gpt4-xl",
                          'flan-t5': "google/flan-t5-xl",
                          'mt0': "bigscience/mt0-large"}
        
    # Exist
    # classes = ['sexist', 'non-sexist'] 
    # Hateval
    classes = ['hate', 'non-hate'] 
    # labels = ['hatespeech', 'non_hatespeech']
    # misocorpus 
    # classes = ['misogyny', 'non-misogyny']
    
    # haternet and hasoc
    # classes = ['hateful', 'non-hateful']
    column_name = 'tweet'
    # column_name = 'text'
    dataset = pd.read_csv(dataset_path)

    # dataset['label'] = dataset['label'].apply(lambda x: 'hate' if 'racist' in x or 'misogyny' in x else x)
    # dataset['label'] = dataset['label'].apply(lambda x: 'non-hate' if x == 'safe' else x)

    dataset = dataset[['tweet', '__split', 'label']]  

    # For hateval 
    dataset['label'] = dataset['label'].apply(lambda x: 'hate' if x == 'hatespeech' else x)
    dataset['label'] = dataset['label'].apply(lambda x: 'non-hate' if x == 'non_hatespeech' else x)
    
    # For hasoc 
    # dataset['label'] = dataset['label'].apply(lambda x: 'hateful' if x == 'HOF' else x)
    # dataset['label'] = dataset['label'].apply(lambda x: 'non-hateful' if x == 'NOT' else x)
   
    # For haternet 
    # dataset['label'] = dataset['label'].apply(lambda x: 'non-hateful' if x == 'non_hateful' else x)
    
    # dataset['label'] = dataset['label'].apply(lambda x: 'non-sexist' if x == 'not sexist' else x)
    
    train_df = dataset[dataset['__split']=='train']
    val_df = dataset[dataset['__split']=='test'] 
    
    val_df_true = val_df[val_df['label'] == classes[0]].head(5)
    val_df_false = val_df[val_df['label'] == classes[1]].head(5)
    
    examples_texts = val_df_true[column_name].tolist() + val_df_false[column_name].tolist()
    examples_labels = val_df_true['label'].tolist() + val_df_false['label'].tolist()
    
    print(examples_texts)
    print(examples_labels)
    
    for k, v in trasformers_models.items(): 
        get_few_shot_prediction(v, val_df, val_df[column_name].tolist(), examples_texts, examples_labels, "hateval-en", k, save_path)
    
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', type=str, default='./hateval-en.csv')
    parser.add_argument('--save_path', type=str, default='./results_hateval-en')
    args = parser.parse_args()
    main(args)
    