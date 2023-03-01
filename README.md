# bios-dutch

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


### Scripts Included:
1. `bionet_unify_people_bios.py` - takes the 'normal' json (one row per biography text) set and generates a unified json (one row per Person - i.e. all texts and metadat are grouped under the same person_id)
2. `find_organization_clusters.py` - Takes a **unified** json and groups all unique people by: 
    * [ORG --> people IDs in which ORG is mentioned under at least one of their biographies]
    * [PersonId --> ORGs mentioned in one of more of the biographies]
    * [PersonId --> Person Metadata Names]
3. `stats_unique_people.py` - It has diverse experiments and statistic extraction of the dataset. It works over the `AllBios_Unified.jsonl` file (find it in surfdrive)
4. `utils/` Folder with auxiliary scripts to compute MetadataComplete operations and statistics