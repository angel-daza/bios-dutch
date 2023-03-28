# bios-dutch

## Get Started

The big files are not included in the repo because of size constraints, so they need to be copied first. There are two kinds of files: 
* `biographynet_{train|test|development}.jsonl`  where each record is a JSON object containing the biography individual text + metadata from the BiographyNet dump `bioport_export_2017-03-10`. Each record has an `id_composed` unique identifier (In the form: `f"{id_person}_{version}"`). These files are the most complete because they include basic NLP information such as tokenization, NER, dependencies, etc...
* `AllBios_Unified.jsonl` where each record is a JSON object per unique person in the database. Each person can have one or more texts associated to it as well as all of the metadata found in the different files associated to the same person ID. In this file train dev and test are all together. See <classes.MetaDataComplete> Objects to understand more about the structure of this file. This file is "lighter" because all the NLP information is omitted. This file was generated with the script `bionet_unify_people_bios.py`

One could load the whole `AllBios_Unified.jsonl` at once and start computing global statistics on it. Each record has an `id_person` unique identifier (the `id_person` is the first component of the `id_composed`). In the case that NLP information is needed, the recommended process is to use the script `bionet_create_individual_files.py` to break down the `biographynet_{train|test|development}.jsonl` into a one-JSON-per-file approach to avoid memory issues (see the script to find out how this can be done). Once all the individual files are generated use the ID `f"{id_person}_{version}"` to load the desired file and obtain the NLP information on demand.


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
2. `find_organization_clusters.py` - Takes a **unified** json and groups all unique people by: 
    * [ORG --> people IDs in which ORG is mentioned under at least one of their biographies]
    * [PersonId --> ORGs mentioned in one of more of the biographies]
    * [PersonId --> Person Metadata Names]
3. `stats_unique_people.py` - It has diverse experiments and statistic extraction of the dataset. It works over the `AllBios_Unified.jsonl` file (find it in surfdrive)
4. `utils/` Folder with auxiliary scripts to compute MetadataComplete operations and statistics including:
    * `bionet_unify_people_bios.py` - takes the 'normal' json (one row per biography text) set and generates a unified json (one row per Person - i.e. all texts and metadat are grouped under the same person_id)
    * `bionet_create_individual_files.py` - functions to manage individual jsons per biography instead of the big json files containing all of them.