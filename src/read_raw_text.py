# -*- coding:utf-8 -*-
# _author_ = xu_qn
# read data in /data/
from src.config import Config
from hashlib import md5
from src.class_list import AnnotatedReview
from src.singular_plural import singularize
import nltk

noun_list = ["NN", "NNP", "NNS", "NNPS"]

def stem_str(str):
    """normalize words (conversion of plural nouns into singular nouns)"""
    tokens= nltk.word_tokenize(str.lower())
    tokens_tag = nltk.pos_tag(tokens)
    # texts = [singularize(i) for i in tokens]
    texts = []
    for i in range(len(tokens)):
        if tokens_tag[i][1] in noun_list:
            texts.append(singularize(tokens[i]))
        else:
            texts.append(tokens[i])
    return " ".join(texts)


def read_raw_file(in_file):
    """read raw reviews, return [hash_code, feature_list, opinion_list, review]"""
    f = open(in_file,"r")
    text_list = []
    for line in f.readlines():
        if len(line.split("##", 1)) == 2:
            list,text = line.split("##",1)
            parts = list.split(",")
            feature_list = []
            opinion_list = []
            for part in parts:
                if '[' in part and len(part[:part.index('[')]) > 0:  # extract reviews with labelled aspect
                    feature_num_word = part[:part.index('[')].strip()
                    if len(feature_num_word.split()) == 1:
                        feature_list.append(part[:part.index("[")].strip())
                if len(feature_list) > 0:
                    text = text.encode("utf-8").decode('utf-8', 'ignore').encode("utf-8").decode('utf-8')
                    hash_code = md5(text.encode("utf-8")).hexdigest()
                    text_list.append(AnnotatedReview(hash_code, feature_list,opinion_list,stem_str(text)))
    return text_list

def write_text(in_file,out_file):
    text_list = read_raw_file(in_file)
    f = open(out_file, "w",encoding="utf-8")
    res_list = []
    for review in text_list:
        if review.hash_code not in res_list:
            f.write(review.hash_code + " ")
            f.write(review.sentence+"\n")
            res_list.append(review.hash_code)
    f.close()


def transfer_format(in_file,out_file):
    text_list = read_raw_file(in_file)
    f = open(out_file,"w")
    res_list = []
    for review in text_list:
        if review.hash_code not in res_list:
            f.write(review.hash_code + " ")
            f.write("[" + ",".join(review.feature_list) + "],")
            f.write("[" + ",".join(review.opinion_list) + "],")
            f.write(review.sentence+"\n")
            res_list.append(review.hash_code)
        else:
            continue
    f.close


if  __name__ == "__main__":
    for file in Config.file_name:
        transfer_format(Config.data_path + file + ".txt", Config.data_path + file + "_id.txt")
        write_text(Config.data_path + file + ".txt", Config.data_path + file + "_text.txt")
    # transfer_format("Apex AD2600 Progressive-scan DVD player.txt","Apex AD2600 Progressive-scan DVD player_id.txt")
    # write_text("Apex AD2600 Progressive-scan DVD player.txt","Apex AD2600 Progressive-scan DVD player_text.txt")
