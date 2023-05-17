from utils.nlp_tasks import run_flair, flair2bio
from flair.data import Sentence
from flair.models import SequenceTagger
from flair.tokenization import SegtokSentenceSplitter
import stanza
from utils.nlp_tasks import run_bert_ner

import json, glob, os
from typing import Dict, Any, List
from collections import defaultdict
from seqeval.metrics import classification_report


import pandas as pd


from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForTokenClassification


# #### ----- AllenNLP Evaluation  ----- #####
# ## This section is fully commented because AllenNLP and Flair conflict with each other
# ## so this code only runs on a separate environment. I included the function in the same script for completenes...
# from allennlp.predictors import Predictor
# import spacy
# from spacy.tokens import Doc
# def evaluate_english_allennlp_wiki_ner():

#     wiki_gold_path = "/Users/daza/Repos/my-vu-experiments/wikigold/"
#     tokens_all = read_wikigold_file(f"{wiki_gold_path}/test.conllu")
#     valid_labels = ['PER', 'ORG', 'LOC', 'MISC']

#     ner_predictor = Predictor.from_path("https://storage.googleapis.com/allennlp-public-models/ner-elmo.2021-02-12.tar.gz")
#     spacy_nlp = spacy.load("en_core_web_sm")
    
#     # Transform the Token-level Data into Document-level & Sentence-level Data
#     sentences = defaultdict(list)
#     for row in tokens_all:
#         sentences[f"{row['document_id']}_{row['sentence_id']}"].append((row['word'], row['label']))
    
#     # Evaluate Sentence-level Data
#     all_bio_predicted, all_bio_gold = [], []
#     all_pred, all_gold = [], []
#     i = 0
#     for seq_id, sequence in sentences.items():
#         words, gold_labels = zip(*sequence)
#         doc = Doc(spacy_nlp.vocab, words)
#         tokenized = [tok.text for tok in doc]
#         output = ner_predictor.predict(doc)
#         predicted_labels = [] 
#         for label in output["tags"]:
#             if label.startswith("U-"):
#                 new_label = label.replace("U-", "B-")
#             elif label.startswith("L-"):
#                 new_label = label.replace("L-", "I-")
#             else:
#                 new_label = label
#             predicted_labels.append(new_label)
#         all_bio_predicted.append(predicted_labels)
#         all_bio_gold.append(list(gold_labels))
#         print(f"{words}\n{tokenized}\n{gold_labels} ({len(gold_labels)})\n{predicted_labels} ({len(predicted_labels)})\n-----")
#         assert len(predicted_labels) == len(gold_labels)
    
#     print("---------- SENTENCE-LEVEL ALLENNLP NER ----------")
#     print(classification_report(all_bio_gold, all_bio_predicted))





def evaluate_english_wiki_ner():

    def bio2spans(tokenized_sentence: List[str], bio_labels: List[str]):
        sentence_entities = []
        ent_tokens, ent_indices, ent_label = [], [], ""
        for ix, bio_tag in enumerate(bio_labels):
            if bio_tag.startswith("B-"):
                if len(ent_label) > 0:
                    sentence_entities.append({'surfaceForm': " ".join(ent_tokens), 'category': ent_label.upper(),'tokenStart': ent_indices[0], 'tokenEnd': ent_indices[-1]+1})
                    ent_indices = []
                    ent_tokens = []
                ent_label = bio_tag[2:]
                ent_tokens.append(tokenized_sentence[ix])
                ent_indices.append(ix)
            elif bio_tag.startswith("I-") or bio_tag.startswith("L-"):
                ent_tokens.append(tokenized_sentence[ix])
                ent_indices.append(ix)
            elif bio_tag == "O" and len(ent_label) > 0:
                sentence_entities.append({'ID': None, 'surfaceForm': " ".join(ent_tokens), 'category': ent_label.upper(), 'tokenStart': ent_indices[0], 'tokenEnd': ent_indices[-1]+1})
                ent_label = ""
                ent_indices = []
                ent_tokens = []
        return sentence_entities


    wiki_gold_path = "/Users/daza/Repos/my-vu-experiments/wikigold/"
    tokens_all = read_wikigold_file(f"{wiki_gold_path}/test.conllu")
    valid_labels = ['PER', 'ORG', 'LOC', 'MISC']

    tagger = SequenceTagger.load('flair/ner-english-large')
    splitter = SegtokSentenceSplitter()
    
    # Transform the Token-level Data into Document-level & Sentence-level Data
    sentences = defaultdict(list)
    documents = defaultdict(list)
    for row in tokens_all:
        sentences[f"{row['document_id']}_{row['sentence_id']}"].append((row['word'], row['label']))
        documents[row['document_id']].append((row['word'], row['label']))
    
    # Evaluate Sentence-level Data
    all_bio_predicted, all_bio_gold = [], []
    all_pred, all_gold = [], []
    i = 0
    for seq_id, sequence in sentences.items():
        words, gold_labels = zip(*sequence)
        flair_obj = run_flair(words, tagger)
        predicted_labels = flair2bio(flair_obj)[0]
        # For Bag-of-Entities Evaluation
        predicted_entities = [ent for ent_list in flair_obj["tagged_ner"] for ent in ent_list if ent["entity"] in valid_labels]
        predicted_entities = set([(ent['text'], ent['entity']) for ent in predicted_entities])
        gold_entities = bio2spans(words, gold_labels)
        gold_entities = set([(ent['surfaceForm'], ent['category']) for ent in gold_entities])
        all_gold += [(seq_id, x[0], x[1]) for x in gold_entities]
        all_pred += [(seq_id, x[0], x[1]) for x in predicted_entities]

        all_bio_predicted.append(predicted_labels)
        all_bio_gold.append(list(gold_labels))
        print(f"{words}\n{gold_labels}\n{predicted_labels}\n{predicted_entities}\n{gold_entities}\n-----")
        flair_obj["bio_pred"] = predicted_labels
        flair_obj["bio_gold"] = gold_labels
        assert len(predicted_labels) == len(gold_labels)
        # json.dump(flair_obj, open(f"{wiki_gold_path}/flair.sent.{seq_id}.json", "w"), indent=2)
        # i += 1
        # if i > 1: break
    
    print("---------- SENTENCE-LEVEL FLAIR NER ----------")
    print(classification_report(all_bio_gold, all_bio_predicted))

    print(f"----- BAG-OF-ENTITIES EVALUATION ------")
    print("\t Label Breakdown:")
    for lbl in valid_labels:
        lbl_gold = [x for x in all_gold if x[2] == lbl]
        lbl_pred = [x for x in all_pred if x[2] == lbl]
        metrics = compute_set_metrics(set(lbl_gold), set(lbl_pred), verbose=False)
        print(f"\t{lbl} --> Precision: {metrics['precision']:.2f}\tRecall: {metrics['recall']:.2f}\tF1 Score: {metrics['f1']:.2f}")
    print("\t----- ALL LABELS ------")
    compute_set_metrics(set(all_gold), set(all_pred), verbose=True)
    
    # # Evaluate Document-level Data
    all_predicted, all_gold = [], []
    i = 0
    for seq_id, document in documents.items():
        words, gold_labels = zip(*document)
        text = " ".join(words)
        gold_entities = bio2spans(words, gold_labels)
        with open(f"{wiki_gold_path}/wikigold_doc_{seq_id}.txt", "w") as f:
            f.write(text)
        with open(f"{wiki_gold_path}/wikigold_doc_{seq_id}.json", "w") as f:
            entities = {"entities": gold_entities}
            json.dump(entities, f, indent=2)
        # flair_obj = run_flair(text, tagger, splitter)
        # predicted_labels = []
        # for pred in flair2bio(flair_obj):
        #     predicted_labels += pred
        # all_predicted.append(predicted_labels)
        # all_gold.append(list(gold_labels))
        # print(f"{words}\n{gold_labels}\n{predicted_labels}\n-----")
        # assert len(predicted_labels) == len(gold_labels)
        # json.dump(flair_obj, open(f"{wiki_gold_path}/flair.doc.{seq_id}.json", "w"), indent=2)
        # i += 1
        # if i > 1: break
    
    # print("---------- DOCUMENT-LEVEL FLAIR NER (Intersection-wise) ----------")
    # print(classification_report(all_gold, all_predicted))


def read_wikigold_file(filepath: str) -> List[Dict[str, Any]]:
    sentence_id, document_id = 0, 0
    doc_start = False
    data = []
    with open(filepath) as f:
        for line in f:
            row = line.strip().split()
            if len(row) > 0:
                if row[1] == '-DOCSTART-':
                    document_id += 1
                    doc_start = True
                else:
                    data.append({'document_id': document_id, 'sentence_id': sentence_id, 'token_id': row[0], 'word': row[1], 'label': row[2]})
            elif doc_start:
                doc_start = False
            else:
                sentence_id += 1
    return data


def evaluate_bionet_dutch_flair_ner():

    dataset_path = "data/biographynet_test.jsonl"
    results_path = "data/nl_flair_predictions_test_all"
    gold_paths = ["data/bionet_gold/biographynet_test_A_gold.json",
                  "data/bionet_gold/biographynet_test_B_gold.json", 
                  "data/bionet_gold/biographynet_test_C_gold.json"]
    
    # OLD_SOURCES = ['weyerman', 'vdaa', 'nnbw', 'elias']

    if not os.path.exists(results_path): os.mkdir(results_path)

    tagger = SequenceTagger.load('flair/ner-dutch-large')
    splitter = SegtokSentenceSplitter()

    # Process NER with FLair (only if needed)
    textid2source = {}
    with open(dataset_path) as corpus:
        for ix, line in enumerate(corpus):
            doc = json.loads(line)
            text_id = doc['id_composed']
            doc_path = f"{results_path}/{text_id}.{doc['source']}.flair.json"
            textid2source[text_id] = doc['source'] 
            if os.path.exists(doc_path):
                print(f"Skipping {text_id} (output already exists)")
                continue
            else:
                print(f"Processing {text_id} ({ix+1})")
                results = run_flair(doc['text_clean'], tagger, splitter)
                results["text_id"] = text_id
                results["source"] = doc['source']
                results["text"] = doc['text_clean']
                json.dump(results, open(doc_path, "w"), indent=2)
    
    # Evaluate Gold vs Flair
    evaluated_ids = evaluate_ner_outputs(gold_paths, results_path, eval_method="bag_of_entities", eval_title="all_flair", 
                         valid_labels=['PER', 'ORG', 'LOC'], textid2source=textid2source)
    return textid2source, evaluated_ids


def evaluate_bionet_dutch_bert_ner(bert_model: str, bert_tokenizer: str, model_label: str):

    dataset_path = "data/biographynet_test.jsonl"
    results_path = f"data/nl_{model_label}_predictions_test_all"
    gold_paths = ["data/bionet_gold/biographynet_test_A_gold.json",
                  "data/bionet_gold/biographynet_test_B_gold.json", 
                  "data/bionet_gold/biographynet_test_C_gold.json"]

    if not os.path.exists(results_path): os.mkdir(results_path)

    model = AutoModelForTokenClassification.from_pretrained(bert_model)
    tokenizer = AutoTokenizer.from_pretrained(bert_tokenizer)

    bert_nlp = pipeline('token-classification', model=model, tokenizer=tokenizer, device=-1)
    stanza_nlp = stanza.Pipeline(lang="nl", processors="tokenize,lemma,pos, depparse, ner", model_dir="/Users/daza/stanza_resources/")

    # Process NER with a BERT-based model (only if needed)
    textid2source = {}
    tokens_per_source = {}
    with open(dataset_path) as corpus:
        for ix, line in enumerate(corpus):
            doc = json.loads(line)
            text_id = doc['id_composed']
            doc_path = f"{results_path}/{text_id}.{doc['source']}.{model_label}.json"
            textid2source[text_id] = doc['source'] 
            if os.path.exists(doc_path):
                print(f"Skipping {text_id} (output already exists)")
            else:
                print(f"Processing {text_id} ({ix+1})")
                results = run_bert_ner(bert_nlp, stanza_nlp, doc['text_clean'])
                results["text_id"] = text_id
                results["source"] = doc['source']
                results["text"] = doc['text_clean']
                json.dump(results, open(doc_path, "w"), indent=2)

                ## For GLobal Stats
                source = doc["source"]
                if source in tokens_per_source:
                    tokens_per_source[source]["documents"] += 1
                    tokens_per_source[source]["sentences"] += len(results["sentences"])
                    for s in results["sentences"]:
                        tokens_per_source[source]["tokens"] += len(s.split())
                else:
                    tokens_per_source[source] = {}
                    tokens_per_source[source]["sentences"] = len(results["sentences"])
                    tokens_per_source[source]["tokens"] = 0
                    tokens_per_source[source]["documents"] = 1
                    for s in results["sentences"]:
                        tokens_per_source[source]["tokens"] += len(s.split())

    # # Evaluate Gold vs BERT
    evaluated_ids = evaluate_ner_outputs(gold_paths, results_path, eval_method="bag_of_entities", eval_title=f"all_{model_label}", lowercase_entities=True,
                         valid_labels=['PER', 'LOC', 'TIME'], textid2source=textid2source)
    
    print(f"\n\n---- INFO CORPUS ----\n\t{tokens_per_source}")
    
    return textid2source, evaluated_ids


def evaluate_old_dutch_flair_ner():
    dataset_path = "data/json"
    results_path = "data/nl_flair_old_original"
    gold_paths = ["data/bionet_gold/biographynet_test_A_gold.json", "data/bionet_gold/biographynet_test_B_gold.json", "data/bionet_gold/biographynet_test_C_gold.json"]
    
    if not os.path.exists(results_path): os.mkdir(results_path)

    tagger = SequenceTagger.load('flair/ner-dutch-large')
    splitter = SegtokSentenceSplitter()

    # Process NER with FLair (only if needed)
    textid2source = {}
    wanted_sources = ["weyerman", "vdaa", "knaw"]
    corpus = get_dutch_bio_texts(dataset_path, wanted_sources)
    for ix, doc in enumerate(corpus):
        doc_path = f"{results_path}/{doc['text_id']}.{doc['source']}.flair.json"
        textid2source[doc['text_id']] = doc['source'] 
        if os.path.exists(doc_path):
            # print(f"Skipping {doc['text_id']} (output already exists)")
            continue
        else:
            print(f"Processing {doc['text_id']} ({ix+1}/{len(corpus)})")
            results = run_flair(doc['text'], tagger, splitter)
            results["text_id"] = doc['text_id']
            results["source"] = doc['source']
            results["text"] = doc['text']
            json.dump(results, open(doc_path, "w"), indent=2)
    
    # Evaluate Gold vs Flair
    evaluated_ids = evaluate_ner_outputs(gold_paths, results_path, eval_method="bag_of_entities", eval_title="Original Old Dutch", 
                         valid_labels=['PER', 'ORG', 'LOC'], textid2source=textid2source)
    return textid2source, evaluated_ids
    

def evaluate_old_dutch_gpt_ner(eval_model: str, eval_lang: str, textid2source: Dict[str, str]):
    """_summary_

    Args:
        eval_model (str): _description_
        eval_lang (str): _description_
        textid2source (Dict[str, str]): _description_
    """
    translations_path = "data/gpt-3/translations"
    gold_paths = ["data/bionet_gold/biographynet_test_A_gold.json", "data/bionet_gold/biographynet_test_A_gold.json", "data/bionet_gold/biographynet_test_A_gold.json"]
    
    results_path = "data"
    results_sub = f"{results_path}/{eval_lang}_flair_old_translated_{eval_model}/"
    if not os.path.exists(results_sub): os.mkdir(results_sub)

    if eval_lang == "nl":
        tagger = SequenceTagger.load('flair/ner-dutch-large')
        splitter = SegtokSentenceSplitter()
    elif eval_lang == "en":
        tagger = SequenceTagger.load('flair/ner-english-large')
        splitter = SegtokSentenceSplitter()

     # Process NER with FLair (only if needed)
    for path in glob.glob(f"{translations_path}/*.txt"):
        basename = os.path.basename(path).strip(".txt")
        person_id, version, task, model, lang = basename.split("_")
        text_id = f"{person_id}_{version}"
        if eval_model == model and eval_lang == lang:
            doc_path = f"{results_sub}/{text_id}.flair.json"
            if os.path.exists(doc_path):
                print(f"Skipping {text_id} (output already exists)")
                continue
            else:
                print(f"Processing {text_id}")
                with open(path) as f: text = f.read()
                results = run_flair(text, tagger, splitter)
                results["text_id"] = text_id
                results["lang"] = lang
                results["source"] = textid2source.get(text_id, 'UNK')
                results["text"] = text
                json.dump(results, open(doc_path, "w"), indent=2)

    # Evaluate Outputs vs Gold Annotation
    evaluate_ner_outputs(gold_paths, results_sub, eval_method="bag_of_entities", eval_title=f"{eval_model}_{eval_lang}", 
                         valid_labels=['PER', 'ORG', 'LOC'], textid2source=textid2source)


def get_dutch_bio_texts(parent_path: str, sources: str):
    data = []
    for source in sources:
        filepath = f"{parent_path}/{source}/*"
        for path in glob.glob(filepath):
            obj = json.load(open(path))
            if ".test." in path:
                partition = "test"
            elif ".development." in path:
                partition = "development"
            else:
                partition = "train"
            data.append({'text_id': obj['id_composed'], 'source': source, 'partition': partition, 'text': obj['text_clean']})
    return data


def evaluate_ner_outputs(gold_paths: List[str], outputs_parent_path: str, eval_method: str, lowercase_entities: bool = False,
                         valid_labels: List[str] = ['PER', 'ORG', 'LOC', 'MISC'], eval_title: str = "", textid2source: Dict = None) -> List[str]:
    """Evaluate NER outputs in different modalities
    Args:
        gold_paths (List[str]): List of Path(s) to the Human Gold Annotations JSON (generated by: BiographyNet/labelstudio_annotation_reader.py)
        outputs_parent_path (str): Path to the directory containing all of the System Outputs in JSON format (generated by functions above in this script)
        eval_method (str): Compute P,R,F1 for the entities with different levels of strictness: "bag_of_entities" | "partial_match" | "strict_match".
        valid_labels (List[str]): List of labels that we want to compare. Defaults to ['PER', 'ORG', 'LOC', 'MISC'] 
        eval_title (str): A label to recognize the evaluation metrics inside a Log
        textid2source (Dict): To break down the results per source
    Raises:
        NotImplementedError: IF eval_method is not recognized we don't know how to evaluate it ...
    """

    # only_these_sources = ['bwsa','dvn', 'rkdartists', 'pdc', 'jews', 'rijksmuseum', 'knaw', 'portraits', 'blnp', 'bwn', 'bwn_1780tot1830', 'bwg', 'wikipedia']

    gold_docs = {}
    for gold_path in gold_paths:
        gold_docs.update(json.load(open(gold_path)))
    all_gold, all_pred = [], []
    total_docs, total_evaluated_docs = 0, 0
    evaluated_ids = []
    for filepath in glob.glob(f"{outputs_parent_path}/*.json"):
        basename = os.path.basename(filepath)
        text_id = basename.split('.')[0]
        total_docs += 1
        # Compute the annotations vs gold (only IF gold available)
        gold_obj = gold_docs.get(text_id, None)
        if gold_obj:
            evaluated_ids.append(text_id)
            total_evaluated_docs += 1
            # Get Gold Info
            gold_entities = [ent for ent in gold_obj['entities'] if ent['category'] in valid_labels]
            # Get Predictions
            doc_obj = json.load(open(filepath))
            predicted_entities = [ent for ent_list in doc_obj["tagged_ner"] for ent in ent_list if ent["entity"] in valid_labels]
            if eval_method == "bag_of_entities":
                # Get Gold Info
                gold_entities = set([(ent['surfaceForm'], ent['category']) for ent in gold_entities if ent['category'] in valid_labels])
                # Get Predictions
                doc_obj = json.load(open(filepath))
                predicted_entities = set([(ent['text'], ent['entity']) for ent in predicted_entities])
                # print(f"----- {text_id} ------")
                # print(f"GOLD: {gold_entities}\nPRED: {predicted_entities}")
                # Compute Doc-level Accuracy
                # metrics = compute_set_metrics(gold_entities, predicted_entities, verbose=True)
                # Accumulate for Global Evaluation
                if lowercase_entities:
                    all_gold += [(text_id, x[0].lower(), x[1]) for x in gold_entities]
                    all_pred += [(text_id, x[0].lower(), x[1]) for x in predicted_entities]
                else:
                    all_gold += [(text_id, x[0], x[1]) for x in gold_entities]
                    all_pred += [(text_id, x[0], x[1]) for x in predicted_entities]
            else:
                raise NotImplementedError("Give a valid eval_method!")

    print(f"----- FINAL {eval_title.upper()} EVALUATION ------")
    save_table = []
    if eval_method == "bag_of_entities":
        print("\t Label Breakdown:")
        for lbl in valid_labels:
            lbl_gold = [x for x in all_gold if x[2] == lbl]
            lbl_pred = [x for x in all_pred if x[2] == lbl]
            metrics = compute_set_metrics(set(lbl_gold), set(lbl_pred), verbose=False)
            print(f"\t{lbl} --> Precision: {metrics['precision']:.2f}\tRecall: {metrics['recall']:.2f}\tF1 Score: {metrics['f1']:.2f}\tSupport: {len(lbl_gold)}")
            save_table.append({"label": lbl, "precision": metrics['precision'], "recall": metrics['recall'], "f1": metrics['f1'], "support": len(lbl_gold)})
        pd.DataFrame(save_table).to_csv(f"{eval_title}_labels.csv")
        if textid2source:
            save_table = []
            print("\t Source Breakdown:")
            sources = set()
            for _, src in textid2source.items():
                sources.add(src)
            for src in list(sources):
                src_gold = [x for x in all_gold if textid2source[x[0]] == src]
                src_pred = [x for x in all_pred if textid2source[x[0]] == src]
                metrics = compute_set_metrics(set(src_gold), set(src_pred), verbose=False)
                print(f"\t{src} --> Precision: {metrics['precision']:.2f}\tRecall: {metrics['recall']:.2f}\tF1 Score: {metrics['f1']:.2f}\tSupport: {len(src_gold)}")
                save_table.append({"source": src, "precision": metrics['precision'], "recall": metrics['recall'], "f1": metrics['f1'], "support": len(src_gold)})
        print("\t----- ALL LABELS ------")
        tot_metrics = compute_set_metrics(set(all_gold), set(all_pred), verbose=True)
        save_table.append({"source": "TOTAL", "precision": tot_metrics['precision'], "recall": tot_metrics['recall'], "f1": tot_metrics['f1'], "support": len(set(all_gold))})
        print(f"Total Documents Evaluated = {total_evaluated_docs} (out of {total_docs})")
        pd.DataFrame(save_table).to_csv(f"{eval_title}_sources.csv")
    
    return evaluated_ids



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

if __name__ == "__main__":

    # evaluate_english_allennlp_wiki_ner()

    # evaluate_english_wiki_ner()

    evaluate_bionet_dutch_flair_ner()

    evaluate_bionet_dutch_bert_ner("surferfelix/ner-bertje-tagdetekst", "GroNLP/bert-base-dutch-cased", "bertje_hist")
    finetuned_gysbert_filepath = "/Users/daza/gysbert_saved_models/EPOCH_2"
    evaluate_bionet_dutch_bert_ner(finetuned_gysbert_filepath, finetuned_gysbert_filepath, "gysbert_hist")

    textid2source, evaluated_ids = evaluate_old_dutch_flair_ner()
    # Distribution of actually annotated documents
    source2textids = {}
    for tid, src in textid2source.items():
        if tid in evaluated_ids:
            if src in source2textids:
                source2textids[src].append(tid)
            else:
                source2textids[src] = [tid]
    print("Baseline Annotated Documents")
    for src, tids in source2textids.items():
        print(f"{src.upper()}: {len(tids)}")
    
    evaluate_old_dutch_gpt_ner("text-davinci-003", "nl", textid2source)
    evaluate_old_dutch_gpt_ner("gpt-3.5-turbo", "nl", textid2source)
    evaluate_old_dutch_gpt_ner("text-davinci-003", "en", textid2source)
    evaluate_old_dutch_gpt_ner("gpt-3.5-turbo", "en", textid2source)