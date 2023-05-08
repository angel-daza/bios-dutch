from dataclasses import dataclass
import json
from typing import Any, Callable, Dict, List, NamedTuple, Tuple, TypeVar, Union
import logging
logger = logging.getLogger(__name__)

from flair.data import Sentence
from flair.models import SequenceTagger
from flair.tokenization import SegtokSentenceSplitter


def flair2bio(flair_obj: Dict) -> List[List[str]]:
    sent_entities = flair_obj["tagged_ner"]
    sentences = flair_obj["sentences"]
    doc_bio_labels = []

    for entities, sentence in zip(sent_entities, sentences):
        sentence = sentence.split()
        sent_bio = ["O"]*len(sentence)
        # Now translate the Entity spans into BIO Notation
        print(entities)
        for ent in entities:
            token_start = ent['start_token']
            token_end =  ent['end_token']
            label = ent['entity']
            for j in range(token_start, token_end):
                if j == token_start:
                    sent_bio[j] = f"B-{label}"
                else:
                    sent_bio[j] = f"I-{label}"
        doc_bio_labels.append(sent_bio)
    return doc_bio_labels


def run_flair(text: Union[str, List[str]], tagger: SequenceTagger, splitter: SegtokSentenceSplitter = None) -> Dict:
    if splitter:
        sentences = splitter.split(text)
        tagger.predict(sentences)
        texts, ner = [], []
        for sentence in sentences:
            tagged_ents = []
            for entity in sentence.get_spans('ner'):
                token_indices = [t.idx for t in entity]
                tagged_ents.append({"text": entity.text, "start": entity.start_position, "end": entity.end_position, "start_token": token_indices[0]-1, "end_token": token_indices[-1],
                                    "entity": entity.get_label("ner").value, "score": entity.get_label("ner").score})
            ner.append(tagged_ents)
            texts.append(sentence.to_tokenized_string())
        return {'tagged_ner':  ner, 'sentences': texts}
    else:
        sentence = Sentence(text)
        tagger.predict(sentence)
        tagged_ents = []
        for entity in sentence.get_spans('ner'):
            token_indices = [t.idx for t in entity]
            tagged_ents.append({"text": entity.text, "start": entity.start_position, "end": entity.end_position, "entity": entity.get_label("ner").value, 
                                "start_token": token_indices[0]-1, "end_token": token_indices[-1], "score": entity.get_label("ner").score})
        return {'tagged_ner': [tagged_ents], 'sentences': [sentence.to_tokenized_string()]}
