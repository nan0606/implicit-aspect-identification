# -*- coding:utf-8 -*-
# _author_ = xu_qn
import xml.etree.ElementTree as ET
from hashlib import md5
from src.class_list import AnnotatedReview
import nltk
from src.singular_plural import *

data_files = [
              "ABSA-15_Laptops_Train_Data.xml",
              "ABSA-15_Restaurants_Train_Final.xml",
              ]



noun_list = ["NN", "NNP", "NNS", "NNPS"]

def stem_str(str):
    tokens = nltk.word_tokenize(str.lower())
    tokens_tag = nltk.pos_tag(tokens)
    # texts = [singularize(i) for i in tokens]
    texts = []
    for i in range(len(tokens)):
        if tokens_tag[i][1] in noun_list:
            texts.append(singularize(tokens[i]))
        else:
            texts.append(tokens[i])
    return " ".join(texts)


def read_2015_xml(xml_file):
    reviews = ET.parse(xml_file).getroot().findall("Review")
    list_text = []
    for sentences in reviews:
        for sentence in sentences.findall("sentences"):
            for sent in sentence.findall("sentence"):
                text = sent.find("text").text
                text = text.encode("utf-8").decode('utf-8', 'ignore').encode("utf-8").decode('utf-8')
                # print(text)
                hash_code = md5(text.encode("utf-8")).hexdigest()
                for aspect in sent.findall("Opinions"):
                    aspect_list = []
                    polarity_list = []
                    for asp in aspect.findall("Opinion"):
                        a = asp.attrib["category"]
                        b = asp.attrib["polarity"]
                        aspect_list.append(a.lower())
                        polarity_list.append(b)
                    list_text.append(AnnotatedReview(hash_code,aspect_list,polarity_list,stem_str(text.strip())))
    file_write = xml_file.replace(".xml", "_id.txt")
    f = open(file_write, "w",encoding='utf-8')
    for review in list_text:
        if review.hash_code not in list_text:
            f.write(review.hash_code + " ")
            f.write("[" + ",".join(review.feature_list) + "],")
            f.write("[" + ",".join(review.opinion_list) + "],")
            f.write(review.sentence+"\n")
        else:
            continue
    f.close

    file_text = xml_file.replace(".xml", "_text.txt")
    f_text = open(file_text, "w", encoding='utf-8')
    res_list = []
    for review in list_text:
        if review.hash_code not in res_list:
            f_text.write(review.hash_code + " ")
            f_text.write(review.sentence + "\n")
            res_list.append(review.hash_code)
        else:
            continue
    f_text.close

if __name__ == "__main__":
    for file in data_files:
        file = "../source/"+file
        read_2015_xml(file)
