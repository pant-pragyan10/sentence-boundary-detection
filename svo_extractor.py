"""
svo_extractor.py
Extracts SVO (Subject-Verb-Object) triplets from sentences using spaCy.
"""
from typing import List, Tuple
import spacy

class SVOExtractor:
    def __init__(self, model: str = 'en_core_web_sm'):
        self.nlp = spacy.load(model)

    def split_sentences(self, text: str) -> List[str]:
        doc = self.nlp(text)
        return [sent.text.strip() for sent in doc.sents]

    def extract_svo(self, sentence: str) -> List[Tuple[str, str, str]]:
        """
        Extract SVO triplets from a sentence.
        Returns a list of (subject, verb, object) tuples.
        """
        doc = self.nlp(sentence)
        svos = []
        for token in doc:
            if token.dep_ == 'ROOT' and token.pos_ == 'VERB':
                subj = [w for w in token.lefts if w.dep_ in ('nsubj', 'nsubjpass')]
                obj = [w for w in token.rights if w.dep_ in ('dobj', 'attr', 'prep', 'pobj', 'obj')]
                if subj and obj:
                    svos.append((subj[0].text, token.text, obj[0].text))
        return svos

    def extract_svo_from_paragraph(self, paragraph: str) -> List[List[Tuple[str, str, str]]]:
        sentences = self.split_sentences(paragraph)
        return [self.extract_svo(sent) for sent in sentences]
