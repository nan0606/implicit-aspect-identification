# -*- coding:utf-8 -*-
# _author_ = xu_qn
#extract dependency relations of each sentence by using Stanford NLP tool
import requests
from src.config import Config
from multiprocessing import Pool
import time

default_config = {
    "server_address": "http://corenlp.run/",
    "ann": "ssplit, tokenize, pos, depparse",
    "outputFormat": "json"
}

s = requests.session()
s.keep_alive = False

def parse_reslut_format(json_res):
    """return ((w1,p1),dep,(w2,p2))"""
    def word_pos(idx,tokens):
        if idx == 0:
            return "ROOT"
        else:
            return "{},{}".format(tokens[idx-1]["word"], tokens[idx-1]["pos"])
    dep_list = []
    for s in json_res["sentences"]:
        tokens = s["tokens"]
        deps = s["enhancedPlusPlusDependencies"]
        for d in deps:
            gov = d["governor"]  # governor
            dep = d["dependent"]  # dependent
            dep_str = "(({}),{},({}))".format(word_pos(gov, tokens), d["dep"], word_pos(dep, tokens))
            dep_list.append(dep_str)
    return ",".join(dep_list)


def parse_one_review(line):
    hash_code, list = line.split(" ", 1)
    _,_, text = list.split("],")
    res = None
    while res is None:
        try:
            res = requests.post(default_config["server_address"],
                                params={"properties":{"annotators":default_config["ann"],"outputFormat":default_config["outputFormat"]}},
                                data=text.strip())
        except Exception as e:
            print("while dep parsing: ", e)
            time.sleep(2)
    try:
        res = res.json(strict=False)
        parsed = parse_reslut_format(res)
        return "{} [{}]".format(hash_code, parsed)
    except Exception as e:
        print("While parsing json:", e)
        return ""



def parse_file(in_file,out_file):
    """save dependency relations"""
    # in_file = open(in_file,"r")
    # lines = in_file.readlines()
    #
    # pool = Pool(processes=8)
    # parsed_text = pool.map(parse_one_review, lines)
    #
    # out_file = open(out_file, "w", encoding="utf-8")
    # for text in parsed_text:
    #     out_file.write(text)
    #     out_file.write('\n')
    # in_file.close()
    # out_file.close()
    in_file = open(in_file, "r")
    out_file = open(out_file, "w", encoding="utf-8")
    for line in in_file.readlines():
        text = parse_one_review(line)
        out_file.write(text+"\n")

    in_file.close()
    out_file.close()

if __name__ == "__main__":
    # for file in Config.file_name:
    #     parse_file(Config.data_path + file + "_id.txt", Config.data_path + file + "_parsed.txt")
    # parse_file("Apex AD2600 Progressive-scan DVD player_id.txt","Apex AD2600 Progressive-scan DVD player_parsed.txt")
    parse_file("../source/ABSA-15_Laptops_Train_Data_id.txt","../source/ABSA-15_Laptops_Train_Data_parsed.txt")
    parse_file("../source/ABSA-15_Restaurants_Train_Final_id.txt","../source/ABSA-15_Restaurants_Train_Final_parsed.txt")
