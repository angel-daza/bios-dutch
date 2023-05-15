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

    # JSON is the only format that saves the whole response metadata. Probably it is a good idea to always keep it...
    # TSV will only work if we explicitly ask in the ChatGPT prompt to dliver the response in that format.
    formats = ["json", "tsv"] 

    # Prompt Tasks
    direct_ner_prompt = "Identify and Label (PERSON, ORGANIZATION, TIME, LOCATION, ARTWORK, MISC) the Named Entities in the following text:\n\n"
    direct_ner_linking_prompt = "Identify and Label (PERSON, ORGANIZATION, LOCATION, MISC) the Named Entities in the following text. Return also the character spans in which these entities appear, a link to their wikipedia page and wikidata ID if it exists. Return the results in TSV Format with Columns: [Entity, Label, Span Start, Span End, Wikipedia Link, WikiData ID]. Text:\n\n"
    translation_prompt = "Translate this into Dutch and American English:\n\n"

    ### -------- CONSTRUCT PROMPTING DATA -------- ###
    wanted_sources = ["weyerman", "vdaa", "knaw"]
    corpus = []
    for src in wanted_sources:
        # translate = get_bio_texts(src, append_prompt=translation_prompt, task_name="translation")
        # direct_ner = get_bio_texts(src, append_prompt=direct_ner_prompt, task_name="direct_ner")
        direct_ner_link = get_bio_texts(src, append_prompt=direct_ner_linking_prompt, task_name="direct_ner_link")
        # corpus += translate
        # corpus += direct_ner
        corpus += direct_ner_link
    
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
    use_model = "gpt-3.5-turbo"

    for obj in corpus:
        if "json" in formats: output_file = f"data/gpt-3/{obj['text_id']}.{use_model}.{obj['task']}.json"
        if "tsv" in formats: output_tsv = f"data/gpt-3/{obj['text_id']}.{use_model}.{obj['task']}.tsv"
        if not os.path.exists(output_file):
            prompt = obj['prompt_text']
            response = send_openai_chat_completion_request(use_model, prompt)
            response_text = response["choices"][0]["message"]["content"]
            obj["response"] = response
            if "json" in formats:
                json.dump(obj, open(output_file, "w"), indent=2)
            if "tsv" in formats:
                with open(output_tsv, "w") as f:
                    f.write(response_text)
        else:
            print(f"Skipping {obj['text_id']}")



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
        temperature=0.1,
        presence_penalty=-1.0,
        messages=[
            {"role": "user", "content": prompt}
        ]
    )
    return response


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

def read_parenthesis_response(response: str) -> set:
    pred_ents = []
    for line in response.split('\n'):
        if len(line) > 1:
            # Active Label
            if "(PERSON)" in line:
                active_label = "PER"
                entity = line.strip(" (PERSON)")
            elif "(ORGANIZATION)" in line:
                active_label = "ORG"
                entity = line.strip(" (ORGANIZATION)")
            elif "(LOCATION)" in line:
                active_label = "LOC"
                entity = line.strip(" (LOCATION)")
            elif "(TIME)" in line:
                active_label = "TIME"
                entity = line.strip(" (TIME)")
            elif "(MISC)" in line:
                active_label = "MISC"
                entity = line.strip(" (MISC)")
            pred_ents.append((entity, active_label))
    return pred_ents


def get_gpt_entities(filepath: str, model: str):
    if model == "text-davinci-003":
        response = json.load(open(filepath))['response']['choices'][0]['text']
    elif model == "gpt-3.5-turbo":
        response = json.load(open(filepath))['response']['choices'][0]['message']['content']
    else:
        raise Exception("Model Unknown!")
    
    print(f"\n---- CHATGPT OUTPUT -----\n{response}\n\n---- END OUTPUT -----\n")
    
    if "-" in response:
        return read_multiline_response(response)
    elif "(PERSON)" in response:
        return read_parenthesis_response(response)
    else:
        return read_comma_separated_response(response)


def evaluate_ner_outputs(model: str, task:str, outputs_parent_path: str, gold_paths: List[str], valid_labels: List[str] = ['PER', 'ORG', 'LOC', 'MISC', 'TIME']):
    gold_docs = {}
    for gold_path in gold_paths:
        gold_docs.update(json.load(open(gold_path)))
    all_gold, all_pred = [], []
    total_docs_evaluated = 0
    for filepath in glob.glob(f"{outputs_parent_path}/*.json"):
        if model in filepath and task in filepath:
            basename = os.path.basename(filepath)
            text_id = basename.split('.')[0]
            # Compute the annotations vs gold (only IF gold available)
            gold_obj = gold_docs.get(text_id, None)
            if gold_obj:
                total_docs_evaluated += 1
                #print(f"----- {text_id} ------")
                # Get Gold Info
                gold_entities = set([(ent['surfaceForm'], ent['category']) for ent in gold_obj['entities'] if ent['category'] in valid_labels])
                # Get Predictions
                predicted_entities = [ent for ent in get_gpt_entities(filepath, model) if ent[1] in valid_labels]
                #print(f"GOLD: {gold_entities}\nPRED: {predicted_entities}")
                # Compute Doc-level Accuracy
                # metrics = compute_metrics(gold_entities, predicted_entities, verbose=True)
                all_gold += [(text_id, x[0], x[1]) for x in gold_entities]
                all_pred += [(text_id, x[0], x[1]) for x in predicted_entities]

    print(f"----- FINAL EVALUATION {model} {task} ------")
    for lbl in valid_labels:
        lbl_gold = [x for x in all_gold if x[2] == lbl]
        lbl_pred = [x for x in all_pred if x[2] == lbl]
        metrics = compute_set_metrics(set(lbl_gold), set(lbl_pred), verbose=False)
        print(f"\t{lbl} --> Precision: {metrics['precision']:.2f}\tRecall: {metrics['recall']:.2f}\tF1 Score: {metrics['f1']:.2f}")
    print("\t----- ALL LABELS ------")
    compute_set_metrics(set(all_gold), set(all_pred), verbose=True)
    print(f"Total Documents Evaluated = {total_docs_evaluated}")


def compute_set_metrics(gold: set, predicted: set, verbose: bool = False) -> Dict[str, Any]:
    match = gold.intersection(predicted)
    error = predicted.difference(gold)
    missed = gold.difference(predicted)
    tp, fp, fn = len(match), len(error), len(missed)
    prec = 0 if tp+fp == 0 else 100*tp/(tp+fp)
    rec = 0 if tp+fn == 0 else 100*tp/(tp+fn)
    f1 = 0 if prec+rec == 0 else 2*(prec*rec)/(prec+rec)
    if verbose:
        print(f"Total Gold = {len(gold)} | Total Predicted = {len(predicted)}")
        print(f"Match = {tp} | Error = {fp} | Missed = {fn}")
        print(f"\tPrecision: {prec:.2f}\n\tRecall: {rec:.2f}\n\tF1 Score: {f1:.2f}")
    return {
        "match": match,
        "error": error,
        "missed": missed,
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
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

    i = 0
    for filepath in glob.glob(f"{outputs_parent_path}/*.json"):
        if model in filepath and task in filepath:
            basename = os.path.basename(filepath)
            text_id = basename.split('.')[0]
            i += 1
            # Compute the annotations vs gold (only IF gold available)
            predicted_tranlsations = get_gpt_translations(filepath, model)
            nl_trans = predicted_tranlsations.get('nl')
            en_trans = predicted_tranlsations.get('en')
            if nl_trans and len(nl_trans) > 1:
                with open(f"{translation_path}/{text_id}_{task}_{model}_nl.txt", "w") as f:
                    f.write(nl_trans)
            else:
                print(f"----- NL {text_id} {i}  -----")
            if en_trans and len(en_trans) > 1:
                with open(f"{translation_path}/{text_id}_{task}_{model}_en.txt", "w") as f:
                    f.write(en_trans)
            else:
                print(f"----- EN {text_id} {i}  -----")


def evaluate_wiki_gold_ner(model: str, valid_labels: List[str]):
    source_docs_path = "data/wikigold"
    direct_ner_prompt = "Identify and Label (PERSON, ORGANIZATION, LOCATION, MISC) the Named Entities in the following text:\n\n"
    openai.api_key = OPENAI_API_KEY
    all_gold, all_pred = [], []
    for filepath in glob.glob(f"{source_docs_path}/*.txt"):
        text_id = os.path.basename(filepath).strip(".txt")
        with open(filepath) as f:
            text = f.read().strip()
        output_file = f"{source_docs_path}/{text_id}.{model}.json"
        obj = {"text_id": text_id}
        if not os.path.exists(output_file):
            print(f"Processing {text_id}")
            prompt = f"{direct_ner_prompt}{text}"
            if model == "text-davinci-003":
                response = send_openai_completion_request(model, prompt)
            elif model == "gpt-3.5-turbo":
                response = send_openai_chat_completion_request(model, prompt)
            else:
                raise NotImplementedError("I don't know how to query that model!")
            obj["prompt"] = prompt
            obj["response"] = response
            json.dump(obj, open(output_file, "w"), indent=2)
        else:
            print(f"########### --- Evaluating {text_id} --- ###########")
            try:
                gpt_ner = get_gpt_entities(output_file, model)
            except:
                gpt_ner = set()
            gold_ner = json.load(open(f"{source_docs_path}/{text_id}.json"))['entities']
            gold_ner = set([(ent['surfaceForm'], ent['category']) for ent in gold_ner])
            print(f"PRED: {gpt_ner}\nGOLD: {gold_ner}\n")
            all_gold += [(text_id, x[0], x[1]) for x in gold_ner if x[1] in valid_labels]
            all_pred += [(text_id, x[0], x[1]) for x in gpt_ner if x[1] in valid_labels]
    
    print(f"----- FINAL EVALUATION {model} ------")
    for lbl in valid_labels:
        lbl_gold = [x for x in all_gold if x[2] == lbl]
        lbl_pred = [x for x in all_pred if x[2] == lbl]
        metrics = compute_set_metrics(set(lbl_gold), set(lbl_pred), verbose=False)
        print(f"\t{lbl} --> Precision: {metrics['precision']:.2f}\tRecall: {metrics['recall']:.2f}\tF1 Score: {metrics['f1']:.2f}")
    print("\t----- ALL LABELS ------")
    metrics = compute_set_metrics(set(all_gold), set(all_pred), verbose=True)

        


def dummy_test(prompt: str, model: str):
    openai.api_key = OPENAI_API_KEY
    if model == "text-davinci-003":
        response = send_openai_completion_request("text-davinci-003", prompt=prompt)
        print(response.keys())
        print(response["usage"]["total_tokens"])
        print(response["choices"][0]["text"])
    else:
        response = send_openai_chat_completion_request("gpt-3.5-turbo", prompt=prompt)
        print(response.keys())
        print(response["usage"]["total_tokens"])
        print(response['choices'][0]['message']['content'])
    
    json.dump(response, open("data/test_response.json", "w"), indent=2)


if __name__ == '__main__':
    # Find Out Encodings for models:
    encoding_gpt3 = tiktoken.encoding_for_model("text-davinci-003")
    print(encoding_gpt3.name)
    encoding_gpt3_5 = tiktoken.encoding_for_model("gpt-3.5-turbo")
    print(encoding_gpt3_5.name)

    ### Examples of "quick tests":
    # dummy_test(prompt="List here the top 10 smartest people in the world right now:\n", model="text-davinci-003")

    # bio = "Edsger Wybe Dijkstra 11 mei 1930-6 augustus 2002 32 Levensbericht door J.H. van Lint Op 6 augustus 2002 overleed op 72-jarige leeftijd Edsger Wybe Dijkstra, sedert 1984 in het buitenland gevestigd gewoon lid van de Sectie Wiskunde. Hiermee verloren wij een van de pioniers van de informatica. Dijkstra werd op 11 mei 1930 geboren te Rotterdam. Zijn vader was een be- kende chemicus, eerst leraar en later directeur van een middelbare school; zijn moeder was een verdienstelijk wiskundige die tot op hoge leeftijd zich met wiskundige problemen bezig hield. Hij bezocht het Gymnasium Erasmianum te Rotterdam waar hij uitblonk in wiskunde en natuurkunde. Enkele eigen- schappen die kenmerkend voor hem waren kwamen in die tijd al aan de dag. Zo streefde hij steeds naar eenvoudige en elegante oplossingen van wiskundi- ge vraagstukken. Zo ook was er zijn liefde voor klassieke muziek en zijn niet onverdienstelijke vaardigheid op de piano. Jaren later zou zijn Boesendorfer het meest markante meubel in zijn huis zijn. Uit idealisme had hij zich voorgenomen om rechten te gaan studeren om la- ter Nederland te vertegenwoordigen bij de Verenigde Naties. Gelukkig brach- ten zijn resultaten bij het eindexamen hem er toch toe om in 1948 naar Leiden te gaan om daar wis- en natuurkunde en later theoretische natuurkunde te studeren. Eerste programmeur in Nederland Hij was zeer actief in het studentenleven en reeds in die tijd hamerde hij op correct taalgebruik, iets waar de meeste studenten zich niet om bekom- meren. Eveneens had hij toen al af en toe een wat uitdagend en provocerend optreden. Later in zijn leven schreef hij aan een vriend dat het helpt als je je af en toe gedraagt of je niet helemaal toerekeningsvatbaar bent! Beslissend voor zijn verdere loopbaan was het feit dat hij aan het eind van zijn derde jaar de gelegenheid kreeg om in Cambridge een cursus van drie weken in het programmeren voor een elektronische rekenmachine te volgen. A. van Wijngaarden, toen directeur van de rekenafdeling van het Mathematisch Centrum te Amsterdam, hoorde hiervan en bood toen Dijkstra een betrekking op het M.C. aan. Zo werd Dijkstra in maart 1952 de eerste Nederlander met het beroep 'programmeur'. Het werd snel duidelijk dat hij in dit gebied verder wilde gaan. Zijn studie in de theoretische natuurkunde werd een formaliteit die hij in 1956 afrondde met het doctoraalexamen. Vanaf 1952 vormde Dijkstra met B.J. Loopstra en C.S. Scholten een drie- manschap dat zich bezig hield met de ontwikkeling en constructie van de elektronische rekenmachines arra ii, ferta en armac. Voor al deze machi- 33 nes werd de software door Dijkstra ontwikkeld. In de periode 1952 - 1956 vond een evolutie plaats in het programmeren, gedeeltelijk doordat de voortdurend toenemende complexiteit van de systemen een beter gestructureerd operating system noodzakelijk maakte, en gedeeltelijk doordat een meer wiskundige benadering van het programmeren een beter inzicht verschafte over de ef- ficientie en correctheid van programma's. In deze periode hield Dijkstra zich ook bezig met het ontwerpen van algoritmen. Hij vond een zeer efficient al- goritme voor de bepaling van het kortste pad tussen twee punten in een graaf. Het is sindsdien bekend als 'Dijkstra's algorithm'. De eerlijkheid gebiedt te vermelden dat hetzelfde algoritme enkele jaren eerder was gepubliceerd door E.F. Moore maar onopgemerkt bleef. Bij de ontwikkeling van de volgende rekenmachine, de xl, werd hij ge- confronteerd met het probleem van nondeterminisme. De oplossing werd het onderwerp van zijn proefschrift. Hij promoveerde in 1959 cum laude met A. van Wijngaarden als promotor. Het werkelijk baanbrekend werk uit de M.C. periode was de ontwikkeling van de programmeertaal algol '60. Samen met J. Zonneveld schreef Dijkstra de eerste compiler voor die taal. algol was een bijzonder heldere programmeertaal die dan ook zeer lang een dominante rol heeft gespeeld. Internationaal expert In 1962 werd Dijkstra benoemd tot hoogleraar in de wiskunde aan de T.H. Eindhoven. Daar werd door een kleine groep onder zijn leiding het the Multiprogramming System voor de X8 ontwikkeld. Dit was het eerste opera- ting system dat uit gekoppelde, expliciet gesynchroniseerde, samenwerkende sequentiele processen bestond. Deze structuur maakte het onder andere mo- gelijk om correctheid te bewijzen. Inmiddels was Dijkstra een internationaal erkend expert op het gebied van de programmatuur en werd hij bij vele sym- posia en congressen gevraagd om een van de hoofdsprekers te zijn. Een van de vele vaak geciteerde uitspraken van Dijkstra is 'Ik houd van wiskunde maar spaar me de mathematen'. Het kwam dan ook aan de the regelmatig voor dat hij verschil van inzicht had met zijn wiskundige colle- ga's. Toen dit in 1967 leidde tot zware kritiek op het eerste onder zijn leiding geschreven proefschrift maakte hij een lange depressie door. Hij kwam die te boven door zich te storten op zijn ideeen over het programmeren en die vast te leggen in zijn Notes on structureel p rog ramming. Binnen korte tijd hebben die zich over de wereld verspreid. 34"
    # prompt = f"Identify and Label (PERSON, ORGANIZATION, LOCATION, MISC) the Named Entities in the following text. Return also the character spans in which these entities appear, a link to their english wikipedia page and wikidata ID if it exists. Return the results in TSV Format with Columns: [Entity, Label, Span, Wikipedia Link, WikiData ID]. Text:\n\n{bio}"
    # dummy_test(prompt, model="gpt-3.5-turbo")
    

    ### Uncomment the following only IF needed:

    # main_for_old_src_experiments()
    # gold_files = [ "/Users/daza/Repos/my-vu-experiments/BiographyNet/data/biographynet_test_B_gold.json", "/Users/daza/Repos/my-vu-experiments/BiographyNet/data/biographynet_test_B_gold.json",
    #               "/Users/daza/Repos/my-vu-experiments/BiographyNet/data/biographynet_test_C_gold.json"]
    # evaluate_ner_outputs(model="text-davinci-003", task="direct_ner", outputs_parent_path="data/gpt-3", gold_paths=gold_files, valid_labels=['PER', 'ORG', 'LOC'])
    # evaluate_ner_outputs(model="gpt-3.5-turbo", task="direct_ner", outputs_parent_path="data/gpt-3", gold_paths=gold_files, valid_labels=['PER', 'ORG', 'LOC'])

    #extract_translations(model="gpt-3.5-turbo", task="translation", outputs_parent_path="data/gpt-3")
    #extract_translations(model="text-davinci-003", task="translation", outputs_parent_path="data/gpt-3")

    evaluate_wiki_gold_ner("gpt-3.5-turbo", valid_labels=["PER", "LOC", "ORG", "MISC"])