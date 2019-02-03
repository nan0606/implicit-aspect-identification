# -*- coding:utf-8 -*-
# _author_ = xu_qn



class AnnotatedReview:


    def __init__(self, hash_code, feature_list, opinion_list, sentence):
        self.hash_code = hash_code
        self.feature_list = feature_list
        self.opinion_list = opinion_list
        self.sentence = sentence


class OpinionWord:

    def __init__(self, word, features_modified=set(), opinions_modified=set(),
                 sentences_from=set(),extracting_rules=set()):
        self.word = str(word)
        self.features_modified = features_modified
        self.opinions_modified = opinions_modified
        self.sentences_from = sentences_from
        self.extracting_rules = extracting_rules

    def get_word(self):
        return self.word

    def get_features(self):
        return self.features_modified


class FeatureWord:

    def __init__(self, word, features_modified=set(),
                 sentences_from=set(),extracting_rules=set()):
        self.word = str(word)
        self.features_modified = features_modified
        self.sentences_from = sentences_from
        self.extracting_rules = extracting_rules

    def get_word(self):
        return self.word
