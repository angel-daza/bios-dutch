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