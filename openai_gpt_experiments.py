import tiktoken
import os, glob, json
from typing import List, Dict, Tuple, Any
import openai
from openai.openai_object import OpenAIObject

with open(".my_openai") as f:
    OPENAI_API_KEY = f.readline().strip()

DAVINCI_PER_TOKEN_COST = 0.02 / 1000
DAVINCI_ENCODING = 'p50k_base'

CHAT_GPT_ENCODING = 'cl100k_base'
CHAT_GPT_PER_TOKEN_COST = 0.002 / 1000


def main_for_old_src_experiments():

    direct_ner_prompt = "Identify and Label (PERSON, ORGANIZATION, TIME, LOCATION, MISC) the Named Entities in the following text:\n\n"
    translation_prompt = "Translate this into Dutch and American English:\n\n"

    ### -------- CONSTRUCT PROMPTING DATA -------- ###
    wanted_sources = ["weyerman", "vdaa", "knaw"]
    corpus = []
    for src in wanted_sources:
        translate = get_bio_texts(src, append_prompt=translation_prompt, task_name="translation")
        direct_ner = get_bio_texts(src, append_prompt=direct_ner_prompt, task_name="direct_ner")
        corpus += translate
        corpus += direct_ner
    
    print("\n### -------- PRICING FOR GPT 3 (Da Vinci) -------- ###")
    tokens_gpt3 = compute_pricing(corpus, DAVINCI_PER_TOKEN_COST, DAVINCI_ENCODING)
    total_cost_gpt3 = tokens_gpt3*DAVINCI_PER_TOKEN_COST
    
    print("\n### -------- PRICING FOR GPT 3.5 (ChatGPT) -------- ###")
    tokens_gpt3_5 = compute_pricing(corpus, CHAT_GPT_PER_TOKEN_COST, CHAT_GPT_ENCODING)
    total_cost_gpt3_5 = tokens_gpt3_5*CHAT_GPT_PER_TOKEN_COST

    print(f"\n--- Pricing TOTAL ---\nTotal Price = {total_cost_gpt3+total_cost_gpt3_5:.2f} USD\n")

    openai.api_key = OPENAI_API_KEY #os.getenv("OPENAI_API_KEY")
    ### ------- DO THE ACTUAL PREDICTIONS USING GPT 3 (THIS SECTION COSTS MONEY!!! )------
    # exit()
    # use_model = "text-davinci-003"
    # for obj in corpus:
    #     output_file = f"data/gpt-3/{obj['text_id']}.{use_model}.{obj['task']}.json"
    #     if not os.path.exists(output_file):
    #         prompt = obj['prompt_text']
    #         response = send_openai_completion_request(use_model, prompt)
    #         obj["response"] = response
    #         json.dump(obj, open(output_file, "w"), indent=2)
    #     else:
    #         print(f"Skipping {obj['text_id']}")
    
    ### ------- DO THE ACTUAL PREDICTIONS USING GPT 3.5 a.k.a ChatGPT (THIS SECTION COSTS MONEY!!! )------
    # exit()
    # use_model = "gpt-3.5-turbo"
    # for obj in corpus:
    #     output_file = f"data/gpt-3/{obj['text_id']}.{use_model}.{obj['task']}.json"
    #     if not os.path.exists(output_file):
    #         prompt = obj['prompt_text']
    #         response = send_openai_chat_completion_request(use_model, prompt)
    #         obj["response"] = response
    #         json.dump(obj, open(output_file, "w"), indent=2)
    #     else:
    #         print(f"Skipping {obj['text_id']}")



def get_bio_texts(source: str, append_prompt: str = "", task_name: str = "my_task"):
    data = []
    filepath = f"data/json/{source}"
    for path in glob.glob(f"{filepath}/*"):
        obj = json.load(open(path))
        if ".test." in path:
            partition = "test"
        elif ".development." in path:
            partition = "development"
        else:
            partition = "train"
        data.append({'text_id': obj['id_composed'], 'source': source, 'partition': partition, 'task': task_name, 'prompt_text': f"{append_prompt}{obj['text_clean']}\n\n"})
    return data


def num_tokens_from_string(string: str, encoding_name: str) -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens


def compute_pricing(texts: List[str], model_token_cost: float, encoding_name: str):
    total_tokens = 0
    for text_obj in texts:
        prompt_tokens = num_tokens_from_string(text_obj['prompt_text'], encoding_name=encoding_name)
        total_tokens += prompt_tokens
    
    print(f"Total Tokens = {total_tokens}")
    print(f"Total Price = {total_tokens*model_token_cost:.2f} USD")
    return total_tokens


def send_openai_completion_request(model_name: str, prompt: str) -> OpenAIObject:
    response = openai.Completion.create(
        model=model_name,
        prompt=prompt,
        temperature=0,
        max_tokens=512,
        top_p=0.5,
        frequency_penalty=0.0,
        presence_penalty=0.0,
        stop=["\"\"\""]
    )
    return response

def send_openai_chat_completion_request(model_name: str, prompt: str) -> OpenAIObject:
    response = openai.ChatCompletion.create(
        model=model_name,
        messages=[
            {"role": "user", "content": prompt}
        ]
    )
    return response


def evaluate_ner_outputs(model: str, task:str, outputs_parent_path: str, gold_path: str):

    def read_multiline_response(response: str) -> set:
        pred_ents = []
        active_label = None
        ents = []
        for line in response.split('\n'):
            if len(line) > 1:
                # Active Label
                if line.startswith("PERSON:"):
                    active_label = "PER"
                elif line.startswith("ORGANIZATION:"):
                    active_label = "ORG"
                elif line.startswith("LOCATION:"):
                    active_label = "LOC"
                elif line.startswith("TIME:"):
                    active_label = "TIME"
                elif line.startswith("MISC:"):
                    active_label = "MISC"
                # "Labeled" Entities
                if line.startswith("-"):
                    ent = line.strip()[2:]
                    if ent not in ["N/A", "None"]:
                        ents.append((ent, active_label))
                pred_ents += ents
        return set(pred_ents)

    def read_comma_separated_response(response: str) -> set:
        pred_ents = []
        active_label = None
        for line in response.split('\n'):
            if len(line) > 1:
                # Active Label
                if line.startswith("PERSON:"):
                    active_label = "PER"
                elif line.startswith("ORGANIZATION:"):
                    active_label = "ORG"
                elif line.startswith("LOCATION:"):
                    active_label = "LOC"
                elif line.startswith("TIME:"):
                    active_label = "TIME"
                elif line.startswith("MISC:"):
                    active_label = "MISC"
                # "Labeled" Entities
                ents = [(e.strip(), active_label) for e in line.split(':')[1].split(",") if e.strip() != 'N/A']
                pred_ents += ents
        return set(pred_ents)

    def get_gpt_entities(filepath: str, model: str):
        if model == "text-davinci-003":
            response = json.load(open(filepath))['response']['choices'][0]['text']
        elif model == "gpt-3.5-turbo":
            response = json.load(open(filepath))['response']['choices'][0]['message']['content']
        else:
            raise Exception("Model Unknown!")
        if "-" in response:
            return read_multiline_response(response)
        else:
            return read_comma_separated_response(response)

    gold_docs = json.load(open(gold_path))
    all_gold, all_pred = [], []
    for filepath in glob.glob(f"{outputs_parent_path}/*.json"):
        if model in filepath and task in filepath:
            basename = os.path.basename(filepath)
            text_id = basename.split('.')[0]
            # Compute the annotations vs gold (only IF gold available)
            gold_obj = gold_docs.get(text_id, None)
            if gold_obj:
                #print(f"----- {text_id} ------")
                # Get Gold Info
                gold_entities = set([(ent['surfaceForm'], ent['category']) for ent in gold_obj['entities'] if ent['category'] in ['PER', 'ORG', 'LOC', 'MISC', 'TIME']])
                # Get Predictions
                predicted_entities = get_gpt_entities(filepath, model)
                #print(f"GOLD: {gold_entities}\nPRED: {predicted_entities}")
                # Compute Doc-level Accuracy
                # metrics = compute_metrics(gold_entities, predicted_entities, verbose=True)
                all_gold += [(text_id, x[0], x[1]) for x in gold_entities]
                all_pred += [(text_id, x[0], x[1]) for x in predicted_entities]

    print(f"----- FINAL EVALUATION {model} {task} ------")
    for lbl in ['PER', 'ORG', 'LOC', 'MISC', 'TIME']:
        lbl_gold = [x for x in all_gold if x[2] == lbl]
        lbl_pred = [x for x in all_pred if x[2] == lbl]
        metrics = compute_set_metrics(set(lbl_gold), set(lbl_pred), verbose=False)
        print(f"\t{lbl} --> Precision: {metrics['precision']:.2f}\tRecall: {metrics['recall']:.2f}\tF1 Score: {metrics['f1']:.2f}")
    print("\t----- ALL LABELS ------")
    compute_set_metrics(set(all_gold), set(all_pred), verbose=True)


def compute_set_metrics(gold: set, predicted: set, verbose: bool = False) -> Dict[str, Any]:
    match = gold.intersection(predicted)
    error = predicted.difference(gold)
    missed = gold.difference(predicted)
    tp, fp, fn = len(match), len(error), len(missed)
    prec = 0 if tp+fp == 0 else 100*tp/(tp+fp)
    rec = 0 if tp+fn == 0 else 100*tp/(tp+fn)
    f1 = 0 if prec+rec == 0 else 2*(prec*rec)/(prec+rec)
    if verbose:
        print(f"\tPrecision: {prec:.2f}\n\tRecall: {rec:.2f}\n\tF1 Score: {f1:.2f}")
    return {
        "match": match,
        "error": error,
        "missed": missed,
        "precision": prec,
        "recall": rec,
        "f1": f1
    }
            

def extract_translations(model: str, task:str, outputs_parent_path: str):

    def get_gpt_translations(path, model):
        translations = {}
        # Read API Response
        if model == "text-davinci-003":
            response = json.load(open(path))['response']['choices'][0]['text']
        elif model == "gpt-3.5-turbo":
            response = json.load(open(path))['response']['choices'][0]['message']['content']
        else:
            raise Exception("Model Unknown!")
        # Parse response
        nl_titles = ["Dutch translation:", "Dutch:", "Nederlands:"]
        en_titles = ["American English translation:", "American English:"]
        response = response.replace("\n", " ")
        for elem in nl_titles+en_titles:
            response = response.replace(elem, "\n$$$$$\n")
        response = response.split("$$$$$")
        if len(response) == 1:
            translations['nl'] = response[0].strip()
        elif len(response) == 2 or 3:
            translations['en'] = response[-1].strip()
            translations['nl'] = response[-2].strip()
        else:
            print(response)
            exit()
        return translations

    translation_path = f"{outputs_parent_path}/translations"
    if not os.path.exists(translation_path): os.mkdir(translation_path)

    for filepath in glob.glob(f"{outputs_parent_path}/*.json"):
        if model in filepath and task in filepath:
            basename = os.path.basename(filepath)
            text_id = basename.split('.')[0]
            print(f"-----  {text_id}  -----")
            # Compute the annotations vs gold (only IF gold available)
            predicted_tranlsations = get_gpt_translations(filepath, model)
            nl_trans = predicted_tranlsations.get('nl')
            en_trans = predicted_tranlsations.get('en')
            if nl_trans and len(nl_trans) > 1:
                with open(f"{translation_path}/{text_id}_{task}_{model}_nl.txt", "w") as f:
                    f.write(nl_trans)
            if en_trans and len(en_trans) > 1:
                with open(f"{translation_path}/{text_id}_{task}_{model}_en.txt", "w") as f:
                    f.write(en_trans)


def dummy_test():
    openai.api_key = OPENAI_API_KEY
    response = send_openai_completion_request("text-davinci-003", prompt="List here the top 10 smartest people in the world right now:\n")
    print(response.keys())
    print(response["usage"]["total_tokens"])
    print(response["choices"][0]["text"])
    json.dump(response, open("cheche.json", "w"), indent=2)


if __name__ == '__main__':
    # Find Out Encodings for models:
    # encoding_gpt3 = tiktoken.encoding_for_model("text-davinci-003")
    # print(encoding_gpt3.name)
    # encoding_gpt3_5 = tiktoken.encoding_for_model("gpt-3.5-turbo")
    # print(encoding_gpt3_5.name)

    # dummy_test()

    # main_for_old_src_experiments()
    
    # evaluate_ner_outputs(model="text-davinci-003", task="direct_ner", outputs_parent_path="data/gpt-3", gold_path="data/biographynet_test_A_gold.json")
    # evaluate_ner_outputs(model="gpt-3.5-turbo", task="direct_ner", outputs_parent_path="data/gpt-3", gold_path="data/biographynet_test_A_gold.json")

    extract_translations(model="gpt-3.5-turbo", task="translation", outputs_parent_path="data/gpt-3")
    extract_translations(model="text-davinci-003", task="translation", outputs_parent_path="data/gpt-3")