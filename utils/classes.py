from dataclasses import dataclass
from typing import List, Dict, Tuple, Union, Optional, Any
from collections import Counter, defaultdict
import re, json


@dataclass 
class IntaviaToken:
    ID: int
    FORM: str
    LEMMA: str
    UPOS: str
    XPOS: str
    HEAD: int
    DEPREL: str
    DEPS: str
    MISC: List[str] = None
    FEATS: Dict[str, str] = None

@dataclass
class IntaviaEntity:
    ID: str
    surfaceForm: str
    category: str
    locationStart: int
    locationEnd: int
    tokenStart: int = None
    tokenEnd: int = None
    method: str = None

    def __eq__(self, other: object) -> bool:
        if type(other) is type(self):
            return self.locationStart == other.locationStart and self.locationEnd == other.locationEnd and self.category == other.category
        else:
            return False
    
    def __hash__(self) -> int:
        return hash((self.locationStart, self.surfaceForm, self.category))

    def span_match(self, other: object):
        if type(other) is type(self):
            return self.locationStart == other.locationStart and self.locationEnd == other.locationEnd
        else:
            return False
    # same Span Start OR same Span End
    def span_partial_match(self, other: object):
        if type(other) is type(self):
            return self.locationStart == other.locationStart or self.locationEnd == other.locationEnd
        else:
            return False
    
    def get_displacy_format(self):
        return {"start": self.locationStart, "end": self.locationEnd, "label": self.category}

@dataclass
class IntaviaTimex:
    ID: str
    surfaceForm: str
    value: str
    category: str
    locationStart: int
    locationEnd: int
    method: str


@dataclass
class IntaviaSentence:
    paragraph: int
    sentence: int
    text: str
    words: List[IntaviaToken]



class IntaviaDocument:
    def __init__(self, intavia_dict: Dict[str, Any]):
        self.text_id: str = intavia_dict['text_id']
        self.text:str = intavia_dict['data']['text']
        self.tokenization: str = intavia_dict['data']['tokenization']
        self.morpho_syntax: List[IntaviaSentence] = [] 
        for sent_obj in intavia_dict['data']['morpho_syntax']:
            tokens = [IntaviaToken(**word_obj) for word_obj in sent_obj['words']]
            sentence = IntaviaSentence(sent_obj['paragraph'], sent_obj['sentence'], sent_obj['text'], tokens)
            self.morpho_syntax.append(sentence)
        self.entities: List[IntaviaEntity] = [IntaviaEntity(**ent) for ent in intavia_dict['data'].get('entities', [])]
        self.time_expressions: List[IntaviaTimex] = [IntaviaTimex(**tim) for tim in intavia_dict['data'].get('time_expressions', [])]
        self.semantic_roles: List[Dict[str, Any]] = intavia_dict['data'].get('semantic_roles', [])
    
    def get_basic_stats(self) -> Dict[str, Any]:
        sentences = []
        verbs, nouns, adjs = [], [], []
        for sent in self.morpho_syntax:
            sentences.append(sent.text)
            for tok in sent.words:
                lemma = tok.LEMMA
                if tok.UPOS == "VERB":
                    verbs.append(lemma)
                elif tok.UPOS in ["NOUN", "PROPN"]:
                    nouns.append(lemma)
                elif tok.UPOS == "ADJ":
                    adjs.append(lemma)
        return {
            'sentences': sentences,
            "top_verbs": Counter(verbs).most_common(10),
            "top_nouns": Counter(nouns).most_common(10),
            "top_adjs": Counter(adjs).most_common(10),
        }
    
    def get_available_methods(self, task_layer: str) -> List[str]:
        if task_layer == "entities":
            return list(set([ent.method for ent in self.entities]))
        elif task_layer == "time_expressions":
            return list(set([ent.method for ent in self.time_expressions]))
        elif task_layer == "semantic_roles":
            return list(set([ent['method'] for ent in self.semantic_roles]))
        else:
            raise ValueError(f"NLP Layer {task_layer} is not a valid layer in the IntaviaDocument") 

    def get_entities(self, methods: List[str] = ['all']) -> List[IntaviaEntity]:
        """_summary_
        Args:
            methods (List[str], optional): Filter entitities according to one or more <methods> | 'all' (everything in the list) | 'intersection' (only entities produced by all models listed in <methods>)
        Returns:
            List[Dict[str, Any]]: The requested list of Entities. Each entitiy is a dictionary with keys: 
                ["ID", "surfaceForm", "category", "locationStart", "locationEnd", "tokenStart", "tokenEnd", "method"]
        """
        if 'all' in methods:
            entities = self.entities
        elif 'intersection' in methods:
            raise NotImplementedError
        else:
            entities = [ent for ent in self.entities if ent.method in methods]
        
        return entities
    
    def get_entity_counts(self, methods: List[str] = ['all'], top_k: int = -1) -> Dict[str, int]:
        entity_src_dict = defaultdict(list)
        entities = self.get_entities(methods=methods)
        for ent_obj in entities:
            entity_src_dict[ent_obj.method].append(ent_obj.category)
        entity_dict = {}
        for src, ents in entity_src_dict.items():
            if top_k > 0:
                entity_dict[src] = Counter(ents).most_common(top_k)
            else:
                entity_dict[src] = Counter(ents).most_common()
        return entity_dict

    def get_entity_category_matrix(self):
        all_methods, all_labels = set(), set()
        entity_info = defaultdict(list)
        for ent in self.entities:
            all_labels.add(ent.category)
            all_methods.add(ent.method)
            entity_info[f"{ent.method}_{ent.category}"].append(ent)
        
        
        sorted_found_labels = sorted(all_labels)
        sorted_found_methods = sorted(all_methods)
        entity_table = [["Category"] + [method for method in sorted_found_methods]]
        for label in sorted_found_labels:
            row = [label]
            for method in sorted_found_methods:
                key = f"{method}_{label}"
                if key in entity_info:
                    ent_counts = len(entity_info[key])
                    row.append(ent_counts)
                else:
                    row.append(0)
            entity_table.append(row)
        
        return entity_table


    def get_confidence_entities(self, mode: str = "spans") -> List[Dict]:
        " mode = 'spans' or 'ents' "
       
        if mode not in ["spans", "ents"]:
            raise NotImplementedError

        entity_agreement = []
        methods = self.get_available_methods("entities")
        max_agreement = len(methods)
        
        if max_agreement == 0: return []

        if mode == "spans":
            charstart2token, charend2token = {}, {}
            for sent in self.morpho_syntax:
                for token in sent.words:
                    charstart2token[token.MISC['StartChar']] = token.ID
                    charend2token[token.MISC['EndChar']] = token.ID

        for ent_obj in self.entities:
            key = f"{ent_obj.surfaceForm}_{ent_obj.locationStart}_{ent_obj.locationEnd}_{ent_obj.category}"
            entity_agreement.append(key)
        entity_agreement = Counter(entity_agreement).most_common()
        entity_confidence_spans = []
        for ent_key, freq in entity_agreement:
            agreement_ratio = freq/max_agreement
            text, start, end, label = ent_key.split("_")
            if agreement_ratio <= 0.3:
                confidence_cat = "LOW"
            elif 0.3 < agreement_ratio <= 0.5:
                confidence_cat = "WEAK"
            elif 0.5 < agreement_ratio <= 0.75:
                confidence_cat = "MEDIUM"
            elif 0.75 < agreement_ratio <= 0.89:
                confidence_cat = "HIGH"
            else:
                confidence_cat = "VERY HIGH"
            
            if mode == "spans":
                token_start = charstart2token.get(int(start))
                token_end = charend2token.get(int(end))
                if type(token_start) == int and type(token_end) == int:
                    entity_confidence_spans.append({"start_token": int(token_start), "end_token": int(token_end) + 1, "label": label})
            else:
                entity_confidence_spans.append({"text": text, "start": int(start), "end": int(end), "label": confidence_cat})
        
        return entity_confidence_spans

    def get_entities_IOB(self) -> List[str]:
        raise NotImplementedError


class Date:
    '''Object to represent dates. Dates can consist of regular day-month-year, but also descriptions (before, after, ca.). Object has attributes for regular parts and one for description, default is empty string.'''

    def __init__( self,  year='YY', month='YY', day='YY', description='', dateInterval = ''):
        self.year = year
        self.month = month
        self.day = day
        self.decade = None
        self.century = None
        self.description = description
        self.interval = dateInterval


    def returnDate(self):
        myDate = self.year + '-' + self.month + '' + self.day
        if self.description:
            myDate += ' (' + self.description + ')'
        return myDate


class Event:
    '''Object that can describe an event (time, place, description)'''
    
    def __init__(self, label, location, date):
        self.label: str = label
        self.location: str = location
        if date and isinstance(date, str):
            date = date.strip()
        elif date and isinstance(date, float):
            date = str(int(date))
        self.date: str = date if date else None
        self.date_tuple: Tuple = (-1, -1, -1)
        self.date_range: Tuple = (-1, -1)
        self.date_is_certain = True
        # The Date Field is too dirty. Here we pack everything in a tuple to later make calculations easier
        if self.date and len(self.date) > 0:
            info_full = re.search(r"(\d{4})-(\d{2})-(\d{2})?", self.date) # Exact Full Date Known
            if info_full: 
                self.date_tuple = (int(info_full.group(1)), int(info_full.group(2)), int(info_full.group(3))) # (1708, 10, 11)
            elif len(self.date) == 4 or len(self.date) == 3: # Only the year is known
                self.date_tuple = (int(self.date), -1, -1) # (1708, )
            elif self.date == '?':
                self.date_is_certain = False
            else:
                info_year_month = re.search(r"(\d{4})-(\d{2})", self.date)
                if info_year_month:
                    self.date_tuple = (int(info_year_month.group(1)), int(info_year_month.group(2)), -1) 
                else:
                    info_range_4 = re.search(r"(\d{4})~(\d{4})", self.date) # Event happened sometime between two years (e.g. 1919~1934)
                    info_range_3 = re.search(r"(\d{3})~(\d{3})", self.date) # Event happened sometime between two years (e.g. 519~534)
                    if info_range_4:
                        self.date_tuple = (int(info_range_4.group(1)), -1, -1) # Arbitrarily choose the first date
                        self.date_range = (int(info_range_4.group(1)),int(info_range_4.group(2)))
                        self.date_is_certain = False
                    elif info_range_3:
                        self.date_tuple = (int(info_range_3.group(1)), -1, -1) # Arbitrarily choose the first date
                        self.date_range = (int(info_range_3.group(1)),int(info_range_3.group(2)))
                        self.date_is_certain = False
                    else:
                        info_year = re.search(r"(\d{4})", self.date)
                        info_year_3 = re.search(r"(\d{3})", self.date)
                        self.date_is_certain = False
                        try:
                            self.date_tuple = (int(info_year.group(1)), -1, -1)
                        except:
                            try:
                                self.date_tuple = (int(info_year_3.group(1)), -1, -1)
                            except:
                                # TODO: Comment the following lines. For now, they are here to explicitly catch "strange" date formats
                                if not any([x.isalpha() for x in self.date]):
                                    print("DATE -->",self.date, len(self.date))
        else:
            self.date_is_certain = False



    def __repr__(self):
        if self.date and self.location:
            return f"date: {self.date}, location: {self.location}"
        elif self.date:
            return f"date: {self.date}"
        elif self.date_tuple:
            return f"{self.date_tuple}"
        else:
            return f"date: UNK"
    
    def to_json(self):
        return {
            'label': self.label,
            'location': self.location,
            'date': self.date
        }

    def setDate(self, date):
        self.date = date

    def setLocation(self, location):
        self.location = location
    
    def getYear(self):
        if self.date_tuple:
            return self.date_tuple[0]
        else:
            return None
    
    def getDate(self):
        return self.date_tuple
    
    def getLocation(self):
        if self.location and len(self.location) > 0:
            return self.location
        else:
            return None


class State:
    '''Object that can describe a state (begin time, end time, place, description)'''
    
    def __init__(self, label, description = '', location = '', beginDate = '', endDate = ''):
        self.label = label
        self.location = location
        self.beginDate = beginDate
        self.endDate = endDate
        self.description = description
    
    def __repr__(self) -> str:
        state_info = f"{self.label.title()}"
        if self.beginDate and self.beginDate != "None":
            state_info += f": from {self.beginDate}"
        if self.endDate and self.endDate != "None":
            state_info += f" until {self.endDate}"
        if self.location and len(self.location) > 0:
            state_info += f" at {self.location}"
        return state_info

    
    def to_json(self):
        return {
            'label': self.label,
            'location': self.location,
            'beginDate': self.beginDate,
            'endDate': self.endDate,
            'description': self.description
        }

    def setBeginDate(self, date):
        self.beginDate = date
    
    def setEndDate(self, date):
        self.endDate = date
    
    def setLocation(self, location):
        self.location = location

    def setDescription(self, description):
        self.description = description
    
    def getLocation(self):
        if self.location and len(self.location) > 0:
            return self.location
        else:
            return None



class MetadataComplete:
    '''Object that represents all available metadata for an individual. All except id number are represented as lists
     WARNING! The .to_json() and .from_json() methods are manual, so new class properties need to be added to this methods as well!
    '''
    
    def __init__(self, idNr):
        self.person_id: str = idNr
        self.versions: List[str] = [] # This allows to 'map back' to the original source of metadata since all lists are equally ordered
        self.sources: List[str] = []
        self.partitions: List[str] = []
        self.names: List[str] = []
        self.births: List[Event] = []
        self.deaths: List[Event] = []
        self.fathers: List[str] = []
        self.mothers: List[str] = []
        self.partners: List[str] = []
        self.educations: List[State] = []
        self.occupations: List[State] = []
        self.genders: List[str] = []
        self.religions: List[str] = []
        self.faiths: List[State] = []
        self.residences: List[State] = []
        self.otherEvents: List[Event] = []
        self.otherStates: List[State] = []
        self.texts: List[str] = []
        self.texts_tokens: List[List[str]] = []
        self.texts_entities: List[List[Dict]] = [] # ent_list_item ~ {'text': 'Amsterdam', 'label': 'LOC', 'start': 70, 'end': 79, 'start_token': 14, 'end_token': 15}
        self.texts_timex: List[List[Dict]] = [] # timex_dict ~ {'tid': 't3', 'type': 'DATE', 'value': '1789-08-11', 'text': '11 Aug. 1789', 'start': 48, 'end': 59}

    def __str__(self) -> str:
        s1 = f" ----- PERSON {self.person_id} -----\n\tName: {self.names}\n\tGender: {self.genders}\n\tBirth: {self.births}\n\tDeath: {self.deaths}\n\tFather: {self.fathers}"
        s2 = f"\n\tMother: {self.mothers}\n\tPartner: {self.partners}\n\tEducation: {self.educations}\n\tOccupation: {self.occupations}\n\tReligion: {self.religions}"
        s3 = f"\n\tN_TEXTS: {len(self.texts)}"
        return s1+s2+s3

    @classmethod
    def from_json(cls, info: Dict):
        person = cls(info['person_id'])
        person.versions = info['versions']
        person.sources = info['sources']
        person.names = info['names']
        person.partitions = info['partitions']
        person.births = [Event(**e) for e in info['births'] if e]
        person.deaths = [Event(**e) for e in info['deaths']if e]
        person.fathers = info['fathers']
        person.mothers = info['mothers']
        person.partners = info['partners']
        person.educations = [State(**s) for s in info['educations'] if s]
        person.occupations = [State(**s) for s in info['occupations'] if s]
        person.genders = info['genders']
        person.religions = info['religions']
        person.faiths = [State(**s) for s in info['faiths'] if s]
        person.residences = [State(**s) for s in info['residences'] if s]
        person.otherEvents = [Event(**e) for e in info['otherEvents'] if e]
        person.otherStates = [State(**s) for s in info['otherStates'] if s]
        person.texts = info['texts']
        person.texts_tokens = info['texts_tokens']
        person.texts_entities = info['texts_entities']
        person.texts_timex = info['texts_timex']
        return person

    def to_json(self):
        return {
            'person_id': self.person_id,
            'versions': self.versions,
            'sources': self.sources,
            'partitions': self.partitions,
            'names': self.names,
            'births': [e.to_json() for e in self.births],
            'deaths': [e.to_json() for e in self.deaths],
            'fathers': self.fathers,
            'mothers': self.mothers,
            'partners': self.partners,
            'educations': [st.to_json() for st in self.educations],
            'occupations': [st.to_json() for st in self.occupations],
            'genders': self.genders,
            'religions': self.religions,
            'faiths': [st.to_json() for st in self.faiths],
            'residences': [st.to_json() for st in self.residences],
            'otherEvents': [e.to_json() for e in self.otherEvents],
            'otherStates': [st.to_json() for st in self.otherStates],
            'texts': self.texts,
            'texts_tokens': self.texts_tokens,
            'texts_entities': self.texts_entities,
            'texts_timex': self.texts_timex
        }

    def addName(self, name):
        self.names.append(name)
    
    def addVersion(self, version):
        self.versions.append(version)
    
    def addSource(self, source):
        self.sources.append(source)
    
    def addPartition(self, partition):
        self.partitions.append(partition)

    def addBirthDay(self, birthEvent):
        self.births.append(birthEvent)
    
    def addDeathDay(self, deathEvent):
        self.deaths.append(deathEvent)
    
    def addFather(self, fatherName):
        self.fathers.append(fatherName)
    
    def defineMother(self, motherName):
        self.mothers.append(motherName)
    
    def definePartner(self, partnerName):
        self.mothers.append(partnerName)
    
    def defineGender(self, gender):
        self.genders.append(gender)
    
    def addReligion(self, religion):
        self.religions.append(religion)
    
    def addOtherEvents(self, otherElist):
        self.otherEvents.append(otherElist)
    
    def addOtherStates(self, otherSlist):
        self.otherStates.append(otherSlist)

    def addText(self, text):
        self.texts.append(text)
    
    def addPreTokenized(self, tokens):
        self.texts_tokens.append(tokens)
    
    def addEntities(self, entities):
        self.texts_entities.append(entities)
    
    def addTimex(self, timex_list):
        self.texts_timex.append(timex_list)
    
    def addEducation(self, edu: State) -> None:
        self.educations.append(edu)
    
    def addFaith(self, faithList: State) -> None:
        self.faiths.append(faithList)
    
    def addOccupation(self, occ: State) -> None:
        self.occupations.append(occ)
    
    def addResidence(self, res: State) -> None:
        self.residences.append(res)
    
    def getName(self, mode: str = 'unique_shortest') -> Union[str, List]:
        """
        Get the names formatted according to the 'mode'. It avoids returning None settings
        Args:
            mode (str): 'unique_shortest' | 'unique_longest' | 'all_names'

        Returns:
            Union[str, List]: String containing a unique name or a List of all possible names mentioned in the metadata
        """
        nicest_name = "UNK"
        if len(self.names) == 0: return nicest_name

        expanded_names_possibilites = []
        for name in self.names:
            if "|" in name:
                subnames = name.split("|")
            elif name != "0":
                subnames = [name]
            else:
                continue
            expanded_names_possibilites.extend(subnames)
        if len(expanded_names_possibilites) == 0: return nicest_name

        ordered_names = sorted(expanded_names_possibilites, key= lambda x: len(x)) # Shortest to Longest

        if mode == 'unique_shortest':
            nicest_name = ordered_names[0]
        elif mode == 'unique_longest':
            nicest_name = ordered_names[-1]
        elif mode == 'all_names':
            clean_names = set()
            for n in ordered_names:
                clean_name = n.title().strip()
                clean_name = re.sub(r"\(\d+\)", "", clean_name)
                clean_name = re.sub(r"\s+", " ", clean_name)
                clean_names.add(clean_name)
            return list(clean_names)
        else:
            raise NotImplementedError
        
        return nicest_name.title()
    
    def getGender(self, method: str = 'most_frequent') -> str:
        """
            method: 'most_frequent' | ?
        """
        if len(self.genders) == 0: return None
        if method == 'most_frequent':
            gender_id = Counter(self.genders).most_common(1)[0][0]
        else:
            raise NotImplementedError
        
        if gender_id == 1.0:
            gender_str = 'male'
        elif gender_id == 2.0:
            gender_str = 'female'
        else:
            gender_str = None
        return gender_str
    
    def getGender_predicted_pronoun_votes(self) -> str:
        masc_votes, fem_votes = 0, 0
        masc_weights = {'hij': 1, 'hem': 1, 'broeder': 1, 'broeder van': 2, 'zn. van': 2, 'zoon van': 2}
        fem_weights = {'zij': 4, 'haar': 2, 'dochter': 2, 'dochter van': 10, 'vrouw': 2}
        masc_patterns = "|".join(masc_weights.keys())
        fem_patterns = "|".join(fem_weights.keys())
        # Get the votes in all available texts
        for text in self.texts:
            init_text = text[:200]
            masc_matches = re.finditer(masc_patterns, init_text, re.IGNORECASE)
            for m in masc_matches:
                t = m.group(0).lower()
                if t in masc_patterns:
                    masc_votes += masc_weights[t]
            fem_matches = re.finditer(fem_patterns, init_text, re.IGNORECASE)
            for m in fem_matches:
                t = m.group(0).lower()
                if t in fem_patterns:
                    fem_votes += fem_weights[t]
        # Return the most voted gender
        if fem_votes == 0 and masc_votes == 0: # Most biographies are from men, so if we know nothing then 'male' is a 'safe choice'...
            gender_str = 'male'
        elif fem_votes >= masc_votes:
            gender_str = 'female'
        else:
            gender_str = 'male'
        # print(f"{self.getName()}\t{len(self.texts)}\t{masc_votes}\t{fem_votes}\t{self.getGender()}\t{gender_str}\t{init_text}")
        return gender_str
    
    def getGender_predicted_first_pronoun(self) -> str:
        # Get the votes in all available texts
        gender_str = None
        for text in self.texts:
            init_toks = text.split()[:30]
            for tok in init_toks:
                clean_tok = tok.strip().replace(",", "").replace(";", "").replace("(", "").replace(".", "").lower()
                if clean_tok in ["hij", "hem", "zoon"]:
                    gender_str = 'male'
                    break
                elif clean_tok in ["zij", "haar", "dochter"]:
                    gender_str = 'female'
                    break
            if gender_str:
                break
        # The most likley gender is male so if there was no info then assign male
        if not gender_str:
            gender_str = 'male'
        # print(f"{self.getName()}\t{len(self.texts)}\t{self.getGender()}\t{gender_str}\t{init_toks}")
        return gender_str


    def getCenturyLived(self) -> Optional[str]:
        """This calculates the century in which a person lived according to the average between birth and death (if both are known). 
         Otherwise, if only one of the dates is known some simple heuristics are used. It Return None for both unknown dates.
         The heuristic is birth + 10 years since their life achievements coulcn't been in their early childhood, like wise death - 10 for the same reason. This is 
         to avoid 'borderline' cases: e.g. a person dying in 1901, should be classified as XIX century. A person born in 1799 should be also XIX century...
         Conflicts still exist but should be less.
        """
        birth_year = self.getBirthDate()[0]
        death_year = self.getDeathDate()[0]
        if birth_year != -1 and death_year != -1:
            year = (birth_year+death_year)/2
        elif birth_year != -1:
            year = birth_year + 10
        elif death_year != -1:
            year = death_year - 10
        else:
            year = -1
        return _get_century(year)

    def getOccupation(self, method: str = 'most_common') -> Optional[str]:
        """ method = 'all_occupations' | 'most_common' | 'stringified_all' """
        return _get_state_info(self.occupations, method)
    
    def getResidence(self, method: str = 'most_common') -> Optional[str]:
        """ method = 'all_residences' |'most_common' | 'stringified_all' """
        return _get_state_info(self.residences, method)

    def getEducation(self, method: str = 'most_common') -> Optional[str]:
        """ method = 'all_educations' |'most_common' | 'stringified_all' """
        return _get_state_info(self.educations, method)
    
    def getReligion(self, method: str = 'all_religions') -> Optional[str]:
        """ method = 'all_religions' | 'stringified_all' """
        if method == 'all_religions':
            return [rel for rel in self.religions if rel]
        elif method == 'stringified_all':
            religions = []
            for rel in self.religions:
                if rel: religions.append(rel)
            if len(religions) == 0: return None
            return ", ".join(religions)
        else:
            raise NotImplementedError
    
    def getTimex(self):
        valid_timex = [t for t in self.texts_timex if t]
        if len(valid_timex) == 0:
            return None
        else:
            return valid_timex

    def getFaith(self, method: str = 'most_common') -> Optional[str]:
        """ method = 'most_common' | 'stringified_all' """
        return _get_state_info(self.faiths, method)


    def getBirthDate(self, method: str = 'most_likely_date') -> Tuple[int, int, int]:
        """ Returns a Tuple(year, month, day) with the date. If it is Fully Unknown it returns None. The default tuple is (-1, -1, -1)
        method (str, optional): 'all_valid_dates' | 'valid_full_dates' | 'valid_years' | 'most_likely_date' | 'stringified_all'
        """
        if method == 'stringified_all' or method == 'all_valid_dates':
            births = set()
            for bev in self.births:
                if bev and bev.date and len(bev.date) > 0:
                    births.add(str(bev))
            if method == 'stringified_all':
                return ", ".join(births)
            else:
                return births
        else:
            return _process_dates_from_events(self.births, method=method)
    
    def birthDate_is_intext(self) -> bool:
        """
            Returns True if the birth_date from metadata is explicitly inside one of the biography texts. Returns False otherwise
        """
        found_in_text = False
        years_from_metadata = [b.date_tuple for b in self.births if b.date_tuple is not None]
        years_from_metadata = [b_tup[0] for b_tup in years_from_metadata]
        for text in self.texts:
            if len(text) == 0: continue
            # Quantify how many times the metadata year is actually in the text
            for birth_year in years_from_metadata:
                if re.search(str(birth_year), text):
                    found_in_text = True
        return found_in_text

    def getBirthDate_baseline1(self) -> int:
        """
            BASELINE 1: Return The first date-like text (\d{4} or \d{3}) found in one of the biography texts
        """
        candidates = []
        birth_date = -1
        for text in self.texts:
            if len(text) == 0: continue
            if len(text) > 200: text = text[:200]
            inferred = re.search(r'\d{4}', text)
            if inferred: 
                candidates.append(int(inferred.group()))
            else:
                inferred = re.search(r'\d{3}', text)
                if inferred: 
                    candidates.append(int(inferred.group()))
        #  Get most common date found in all texts
        if len(candidates) > 0:
            proposed_dates = Counter(candidates).most_common() 
            birth_date = proposed_dates[0][0]
        return birth_date
    
    def getBirthDate_baseline2(self) -> int:
        """
            BASELINE 1: Return The first date-like text (\d{4} or \d{3}) found in one of the biography texts
        """
        keywords = '|'.join(['geb.', 'geboren', 'Geb.', 'Geboren'])
        candidates = []
        birth_date = -1
        for text in self.texts:
            if len(text) == 0: continue
            # Get a Narrow window around one of the provided keywords ...
            keyword_match = re.search(keywords, text)
            chargram_window = 30
            if keyword_match:
                m_start, m_end = keyword_match.span()
                start = m_start - chargram_window if m_start > chargram_window else 0 # Maybe the window should only go forward! do not move starting point...
                end = m_end + chargram_window if m_end - chargram_window < len(text) else len(text)
                text = text[start:end]
                # Search for Year as usual ...
                inferred = re.search(r'\d{4}', text)
                if inferred: 
                    candidates.append(int(inferred.group()))
                else:
                    inferred = re.search(r'\d{3}', text)
                    if inferred: 
                        candidates.append(int(inferred.group()))
        #  Get most common date found in all texts
        if len(candidates) > 0:
            proposed_dates = Counter(candidates).most_common() 
            birth_date = proposed_dates[0][0]
        return birth_date

    def getBirthDate_from_timexps(self) -> int:
        """
            BASELINE TIMEX: Return The first HeidelTime year identified in text
        """
        birth_year = -1
        if not self.getTimex(): return birth_year
        first_ones = [tim[0] for tim in self.texts_timex if len(tim) > 0]
        if len(first_ones) == 0: 
            return birth_year

        candidates = []
        for first_one in first_ones:
            inferred = re.search(r'\d{4}', first_one['value'])
            if inferred:
                candidates.append(int(inferred.group()))
            else:
                inferred = re.search(r'\d{3}', first_one['value'])
                if inferred: 
                    candidates.append(int(inferred.group()))
        #  Get most common date found in all texts
        if len(candidates) > 0:
            proposed_dates = Counter(candidates).most_common() 
            birth_year = proposed_dates[0][0]
        return birth_year
    
    def getDeathDate(self, method: str = 'most_likely_date') -> str:
        """ Returns a Tuple(year, month, day) with the date. If it is Fully Unknown it returns None. The default tuple is (-1, -1, -1)
        method (str, optional):  'all_valid_dates' | 'valid_full_dates' | 'valid_years' | 'most_likely_date' | 'stringified_all'
        """
        if method == 'stringified_all' or method == 'all_valid_dates':
            deaths = set()
            for dev in self.deaths:
                if dev and dev.date and len(dev.date) > 0:
                    deaths.add(str(dev))
            if method == 'stringified_all':
                return ", ".join(deaths)
            else:
                return deaths
        else:
            return _process_dates_from_events(self.deaths, method=method)
    
    def getBirthPlace(self) -> str:
        if self.births and len(self.births) > 0:
            known_locations = [ev.getLocation() for ev in self.births if ev.getLocation()]
            if len(known_locations) > 0:
                return Counter([ev.getLocation() for ev in self.births if ev.getLocation()]).most_common(1)[0][0]
            else:
                return None
        else:
            return None

    def getDeathPlace(self) -> str:
        if self.deaths and len(self.deaths) > 0:
            known_locations = [ev.getLocation() for ev in self.deaths if ev.getLocation()]
            if len(known_locations) > 0:
                return Counter([ev.getLocation() for ev in self.deaths if ev.getLocation()]).most_common(1)[0][0]
            else:
                return None
        else:
            return None
    
    def getEntityMentions(self, entity_label: str, text_ix: int = -1) -> List[str]:
        ents = []
        if text_ix < 0:
            for ent_list in self.texts_entities:
                for ent in ent_list:
                    if ent and ent['label'] == entity_label:
                        ents.append(ent['text'])
        elif text_ix < len(self.texts):
            for ent in self.texts_entities[text_ix]:
                if ent and ent['label'] == entity_label:
                        ents.append(ent['text'])
        return ents

    def getRelatedMetadataPlaces(self) -> List[str]:
        ' All events and states metadata can have locations associated to them. Here we return everything we find...'
        # Events: self.births, self.deaths, self.otherEvents
        # States: self.educations, self.occupations, self.faiths, self.residences, self.otherStates
        places = set()
        for ev in self.births:
            places.add(ev.getLocation())
        for ev in self.deaths:
            places.add(ev.getLocation())
        for ev in self.otherEvents:
            places.add(ev.getLocation())
        for st in self.educations:
            places.add(st.getLocation())
        for st in self.occupations:
            places.add(st.getLocation())
        for st in self.faiths:
            places.add(st.getLocation())
        for st in self.residences:
            places.add(st.getLocation())
        for st in self.otherStates:
            places.add(st.getLocation())
        places.remove(None)
        return list(places)


# Auxiliary Functions
def _process_dates_from_events(date_events: List[Event], method: str) -> Tuple[int, int, int]:
        """
            method: 'valid_full_dates' | 'valid_years' | 'most_likely_date'
        """

        valid_dates = set()
        for event in date_events:
            if event.date_tuple != (-1, -1, -1): 
                valid_dates.add(event.date_tuple)

        if method == 'valid_full_dates':
            valid_full = set([d for d in valid_dates if d[1] != -1 and d[2] != -1])
            return list(valid_full)
        elif method == 'valid_years':
            valid_years = set([d[0] for d in valid_dates])
            return list(valid_years)
        elif method == 'most_likely_date':
            valid_years = [d[0] for d in valid_dates if d[0] > 0]
            valid_months = [d[1] for d in valid_dates if d[1] > 0]
            valid_days = [d[2] for d in valid_dates if d[2] > 0]
            if len(valid_years) > 0:
                most_repeated_year = Counter(valid_years).most_common(1)[0][0]
                if len(valid_months) > 0:
                    most_repeated_month = Counter(valid_months).most_common(1)[0][0]
                else:
                    most_repeated_month = -1
                if len(valid_days) > 0:
                    most_repeated_day = Counter(valid_days).most_common(1)[0][0]
                else:
                    most_repeated_day = -1
                my_date = (int(most_repeated_year), int(most_repeated_month), int(most_repeated_day)) # Ensemble a Full-Date with the most frequent data
                return my_date
            else:
                return (-1, -1, -1)
        else:
            print(f"Invalid Date Processing Method {method}")
            raise NotImplementedError


def _get_state_info(states: List[State], method: str):
    states_str = []
    for st in states:
        if st: states_str.append(st.label.title())
    if len(states_str) == 0:
        return None
    else:
        if method.startswith('all_'): # Return all NON None values
            return states_str
        elif method == 'most_common':
            return Counter(states_str).most_common(1)[0][0]
        elif method == 'stringified_all':
            return ", ".join(states_str)
        else:
            raise NotImplementedError


def _get_century(year: int):
    'Return a Century String according to the year Int'
    century = ''
    if year == -1:
        century = None
    elif year < 0:
        century = 'OLD'
    elif 0 < year <= 1000:
        century = 'X (or less)'
    elif 1000 < year <= 1100:
        century = 'XI'
    elif 1100 < year <= 1200:
        century = 'XII'
    elif 1200 < year <= 1300:
        century = 'XIII'
    elif 1300 < year <= 1400:
        century = 'XIV'
    elif 1400 < year <= 1500:
        century = 'XV'
    elif 1500 < year <= 1600:
        century = 'XVI'
    elif 1600 < year <= 1700:
        century = 'XVII'
    elif 1700 < year <= 1800:
        century = 'XVIII'
    elif 1800 < year <= 1900:
        century = 'XIX'
    elif 1900 < year <= 2000:
        century = 'XX'
    elif year > 2000:
        century = 'XXI'
    else:
        century = None
    return century

