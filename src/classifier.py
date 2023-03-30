import requests
import json

from transformers import AutoTokenizer, AutoModelForSequenceClassification, TextClassificationPipeline
import spacy
import en_core_web_sm


class Classifier:
    def __init__(self, metamap_url="https://ii.nlm.nih.gov/metamaplite/rest/annotate"):
        self.metamap_url = metamap_url
        self.spacy_tokenizer = en_core_web_sm.load()
        self.bert_tokenizer = AutoTokenizer.from_pretrained("bvanaken/clinical-assertion-negation-bert")
        self.model = AutoModelForSequenceClassification.from_pretrained("bvanaken/clinical-assertion-negation-bert")
        self.negation_classifier = TextClassificationPipeline(model=self.model, tokenizer=self.bert_tokenizer,
                                                              top_k=None)

    # call the metamap api to extract list of mentioned symptoms from clinical text
    def extract_symptoms(self, text):

        # see https://lhncbc.nlm.nih.gov/ii/tools/MetaMap/run-locally/MetaMapLiteReST.html for api options
        payload = [('inputtext', text),
                   ('docformat', 'freetext'),
                   ('resultformat', 'json'),
                   ('sourceString', 'all'),
                   ('semanticTypeString', 'sosy')]  # sosy = sign or symptom
        headers = {'Accept': "text/plain"}
        r = requests.post(self.metamap_url, payload, headers=headers)
        return json.loads(r.text)

    # given a text tokenized with spacy and an index, return the sentence that index is in
    def extract_sentence(self, index, tokenized_text):
        for sent in tokenized_text.sents:
            if index >= sent.start_char and index < sent.end_char:
                return sent.text
        return ""

    # given an symptom in clinical text, check if is being mentioned positivly/negativly
    def negation_check(self, text, matchedtext, start):
        tokens = self.spacy_tokenizer(text)
        sentence = self.extract_sentence(start, tokens)

        # when inputted into the BERT model, the symptom mentioned must be labeled by "[entity]" tags
        model_input = sentence.replace(matchedtext, f'[entity] {matchedtext} [entity]')

        classifier_output = self.negation_classifier(model_input)

        # compare positive and negative probability to return
        positive_prob = 0
        negative_prob = 0
        for i in range(3):
            if classifier_output[0][i]['label'] == 'PRESENT':
                positive_prob = classifier_output[0][i]['score']
            if classifier_output[0][i]['label'] == 'ABSENT':
                negative_prob = classifier_output[0][i]['score']

        return positive_prob > negative_prob

    def classify(self, patient_file_text, cui, metamap_json=None):

        if metamap_json == None:
            metamap_json = self.extract_symptoms(patient_file_text)

        # find all mentions of the desired symtom by their cui (UMLS taxonomy id)
        found_mentions = []
        for symptom in metamap_json:
            for eval in symptom['evlist']:
                if eval["conceptinfo"]["cui"] == cui:
                    found_mentions.append(eval)

        if len(found_mentions) == 0:
            return ""

        # in case there are multiple mentions of the symptom, we check if there are more
        # positive or negative cases.
        positive_count = 0
        for symptom_mention in found_mentions:
            classifier_output = self.negation_check(patient_file_text, symptom_mention['matchedtext'],
                                                    symptom_mention['start'])
            if classifier_output:
                positive_count += 1

        if positive_count > (len(found_mentions) // 2):
            return "Positive"
        else:
            return "Negative"