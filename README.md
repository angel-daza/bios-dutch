# bios-dutch


## Get Started

1. Create a new environment from scratch and install the required modules:

```
conda create -n bios-dutch python=3.10
conda activate bios-dutch
pip install -r requirements.txt
```

2. The big files are not included in the repo because of size constraints, so they need to be copied first into the `data` folder (if it doesn't exist, create one). The Seed Data (big) files that should be included inside the 'data/seed_data' folder are (if the folder does not exist, please create it):
    * `AllBios.jsonl`
    * `biographynet_train.jsonl`
    * `biographynet_development.jsonl`
    * `biographynet_test.jsonl`

3. Generate the `AllBios_Unified.jsonl` file by running: 

```
python utils/bionet_unify_people_bios.py "data/seed_data/AllBios.jsonl" "data/All_Bios_Unified.jsonl"
```

4. There are two kinds of files: 
    * `biographynet_{train|test|development}.jsonl`  where each record is a JSON object containing the biography individual text + metadata from the BiographyNet dump `bioport_export_2017-03-10`. Each record has an `id_composed` unique identifier (In the form: `f"{id_person}_{version}"`). These files are the most complete because they include basic NLP information such as tokenization, NER, dependencies, etc...
    * `AllBios_Unified.jsonl` where each record is a JSON object per unique person in the database. Each person can have one or more texts associated to it as well as all of the metadata found in the different files associated to the same person ID. In this file train dev and test are all together. See <classes.MetaDataComplete> Objects to understand more about the structure of this file. This file is "lighter" because all the NLP information is omitted. This file was generated with the script `utils/bionet_unify_people_bios.py`

One could load the whole `AllBios_Unified.jsonl` at once and start computing global statistics on it. Each record has an `id_person` unique identifier (the `id_person` is the first component of the `id_composed`). In the case that NLP information is needed, the recommended process is to use the script `utils/bionet_create_individual_files.py` (see step 5).

5. (OPTIONAL) - Skip to step **6.** if you prefer the InTaVia Format. Or... Create **individual biography json** files with `utils/bionet_create_individual_files.py`. This will create a file for each biography in the dataset. All the files will be saved in `data/json/bionet`. It can also break down the `biographynet_{train|test|development}.jsonl` files. This one-JSON-per-file approach is preferred to bypass the need of MongoDB and avoid memory issues. Once all the individual files are generated use the ID `f"{id_person}_{version}"` to load the desired file and obtain the NLP information on demand.

6. (OPTIONAL) - Convert files to **InTaVia Format** by running `python utils/bionet_json_to_intavia_json.py`. This only requires the `AllBios.json` file as input. It will create individual json files inside source subfolders. Each json is one biography following the [InTaVia Format](https://github.com/InTaVia/nlp-pipelines/blob/main/schema/intavia-wp4-json-template-draft.json). Recommended run:

```
python3 utils/bionet_json_to_intavia_json.py files "data/seed_data/AllBios.jsonl" "flask_app/backend_data/intavia_json"
```

The recommended location is `flask_app/backend_data/intavia_json` so the flask_app has no problem finding the static files in the disk.

## Flask Web App

The flask Web App in this repository works with two formats:
* MetadataComplete - To visualized unified information per Person. Prerequisite: Generate Unified Bios file (Step 3)
* IntaviaDocument - To visualize, compare and evaluate NLP Related Tasks. Prerequisite: Generate Individual Intavia Files (Step 6)

Before running the app:

1. We have to pre_compute values and create DataFrames that will emulate a Database. Inside `create_related_datasets.py` there are functions to create such data. Make sure to properly set the paths for `INPUT_JSON` and `FLASK_SEARCHABLE_DF` variables.

2. If you executed step 1 then there is access already to the StanzaNLP outputs (since this was used to preprocess the whole corpus). If you want/need to add more NLP outputs (for example more NER predictions), or gold annotations, then you should use the `add_nlp_layers.py` script. It is very important to set the global variable `JSON_BASEPATH` to point to the Intavia JSON root folder (the recommended location is `flask_app/backend_data/intavia_json`). The script reads the Intavia files, and overrides them so make sure you are running the desired output.

## Useful Info:

### Keys of Non unified JSONL Files:
```
dict_keys(['id_person', 'version', 'id_composed', 'source', 'name', 'partition', 'nlp_processor', 'meta_keys', 'had_html', 'original_path', 
            'birth_pl', 'birth_tm', 'baptism_pl', 'baptism_tm', 'death_pl', 'death_tm', 'funeral_pl', 'funeral_tm', 'marriage_pl', 'marriage_tm', 'gender', 
            'category', 'father', 'mother', 'partner', 'religion', 'educations', 'faiths', 'occupations', 'residences', 'text_clean', 'text_original', 
            'text_tokens', 'text_token_objects', 'text_sentences', 'text_entities', 'text_timex', 'tokens_len', 'meta_len', 'hhucap_annotations'])
```

### Keys of Unified JSONL Files:
```
dict_keys(['person_id', 'versions', 'sources', 'partitions', 'names', 'births', 'deaths', 'fathers', 'mothers', 'partners', 'educations', 'occupations', 'genders', 'religions', 'faiths', 'residences', 'otherEvents', 'otherStates', 'texts', 'texts_tokens', 'texts_entities', 'texts_timex'])
```

The unified files can be loaded in Python <classes.MetaDataComplete> Objects. They 'centralize' all of the information tied to the same person ID. Each field contains a list, where the element of the list can be mapped back to the source and complete text_id if wanted. The fields text_entities, text_tokens, sources, etc... are still separated per individual text file (see find_organization_clusers.py for an example of how to use it)


## Scripts Included:
1. `add_nlp_layers.py` - Contains functions to add gold information and NLP output layers into the existing InTavia JSON files. All NLP related info should be added through this script. Currently it can add the gold NER annotations for the test set, and Flair NER layer. The script is internally and closely related to `utils/nlp_tasks.py`.
2. `find_organization_clusters.py` - Takes a **unified** json and groups all unique people by: 
    * [ORG --> people IDs in which ORG is mentioned under at least one of their biographies]
    * [PersonId --> ORGs mentioned in one of more of the biographies]
    * [PersonId --> Person Metadata Names]
3. `stats_unique_people.py` - It has diverse experiments and statistic extraction of the dataset. It works over the `AllBios_Unified.jsonl` file (find it in surfdrive)
4. `utils/` Folder with auxiliary scripts to compute MetadataComplete operations and statistics including:
    * `bionet_unify_people_bios.py` - takes the 'normal' json (one row per biography text) set and generates a unified json (one row per Person - i.e. all texts and metadat are grouped under the same person_id)
    * `bionet_create_individual_files.py` - functions to manage individual jsons per biography instead of the big json files containing all of them.
    * `classes.py` - contains schemas and object definitions that wrap data structures used in this project (Biography, Token, Event, etc...). These classes are used by several sripts in this repo.
    * `nlp_tasks.py` - contains wrappers to external NLP libraries. This wrappers can be called from other scripts such as the `add_nlp_layers.py`.
5. `create_related_datasets.py` - [TODO]
6. `flask_app/` Folder that self-contains the necessary files to run the WebApp that is a search engine and NLP visualizer of the biographies.