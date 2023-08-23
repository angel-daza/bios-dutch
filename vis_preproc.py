import os
import json
from utils_general import FLASK_ROOT, INTAVIA_JSON_ROOT


def process_json_files(root_folder):
    hard_coded_methods = ['gpt-3.5-turbo', 'human_gold', 'stanza_nl', 'flair/ner-dutch-large_0.12.2', 'gysbert_hist_fx_finetuned_epoch2', 'xlmr_ner_']
    method_total = {m : 0 for m in hard_coded_methods}

    hard_coded_categories = ['', 'PERSON', 'LOCATION', 'ORGANIZATION', 'TIME', 'LOC', 'ARTWORK', 'NUMBER', 'ORG', 'MISC', 'PER', 'DATE']
    category_total = {m : 0 for m in hard_coded_categories}

    category_per_method_total = {
        m : {c : 0 for c in hard_coded_categories} for m in hard_coded_methods 
    }

    method_bio_sort = {m : {} for m in hard_coded_methods}

    category_per_id = {c : {} for c in hard_coded_categories}

    per_id = {}
    id_gold = {}
    
    for folder_name, subfolders, filenames in os.walk(root_folder):
        for filename in filenames:
            if filename.endswith('.json'):
                file_path = os.path.join(folder_name, filename)
                with open(file_path, 'r') as json_file:
                    json_data = json.load(json_file)
                    entities = json_data["data"]["entities"]

                    textid = json_data["text_id"]

                    per_id[textid] = {m : 0 for m in hard_coded_methods if m != "human_gold"}
                    id_gold[textid] = 0

                    for ent in entities:
                        method = ent["method"]
                        method_total[method] += 1

                        if textid in method_bio_sort[method].keys():
                            method_bio_sort[method][textid] += 1
                        else:
                            method_bio_sort[method][textid] = 1


                        category = ent["category"]

                        category_total[category] += 1

                        category_per_method_total[method][category] += 1

                        if textid in category_per_id[category].keys():
                            category_per_id[category][textid] += 1
                        else:
                            category_per_id[category][textid] = 1

                        if method != "human_gold":
                            per_id[textid][method] += 1
                        else:
                            id_gold[textid] += 1
                        

                        
    ids_sorted_per_method = {m: sorted(method_bio_sort[m].keys(), key=lambda x: method_bio_sort[m][x], reverse=True) for m in hard_coded_methods}
    ids_sorted_per_category = {c: sorted(category_per_id[c].keys(), key=lambda x: category_per_id[c][x], reverse=True) for c in hard_coded_categories}
    
    max_dist_per_id = {}

    for id in per_id.keys():
        values = per_id[id]
        gold = id_gold[id]
            
        max_id = max(values, key=values.get)
        min_id = min(values, key=values.get)
        max_count = values[max_id]
        min_count = values[min_id]

        if gold != 0:

            gold_distance = max(abs(gold - max_count), abs(gold - min_count))
            max_gold_dist_method = max_id if abs(gold - max_count) > abs(gold - min_count) else min_id

            gold_variance = [val for val in values.values()]
            gold_variance = [(val - gold)**2 for val in gold_variance]
            gold_variance = sum(gold_variance)/(len(hard_coded_methods) - 1)

        else:
            gold_distance = -1
            gold_variance = -1
            max_gold_dist_method = -1

        max_dist_per_id[id] = {
            "distance" : max_count - min_count,
            "max" : max_id,
            "min" : min_id,
            "gold_distance" : gold_distance,
            "gold_variance" : gold_variance,
            "max_to_gold" : max_gold_dist_method
        }

    dist_sorted_ids = sorted(max_dist_per_id.keys(), key=lambda id_: max_dist_per_id[id_]["distance"])
    gold_dist_sorted_ids = sorted(max_dist_per_id.keys(), key=lambda id_: max_dist_per_id[id_]["gold_distance"])

    d = {
        "method_total" : method_total,
        "category_total" : category_total,
        "category_per_method_total" : category_per_method_total,
        "method_per_id" : method_bio_sort,
        "ids_sorted_per_method": ids_sorted_per_method,
        "category_per_id" : category_per_id,
        "ids_sorted_per_category" : ids_sorted_per_category,
        "per_id" : per_id,
        "id_gold" : id_gold,
        "max_dist_per_id" : max_dist_per_id,
        "dist_sorted" : dist_sorted_ids,
        "gold_dist_sorted" : gold_dist_sorted_ids
    }

    return d

root_folder_path = f"{FLASK_ROOT}/intavia_json/"
d = process_json_files(root_folder_path)

processed_file_path = f"{FLASK_ROOT}/biographies/statistics.json"
with open(processed_file_path, "w") as output:
    json.dump(d, output, sort_keys=False, indent=4)
