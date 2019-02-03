# -*- coding:utf-8 -*-
# _author_ = xu_qn
# extract sapect and opinion words by DP
from src.config import Config
from src.class_list import OpinionWord, FeatureWord
import src.propagation_rules as rule


def extract_depency(file_name):
    dependency_list = {}
    for line in file_name.readlines():
        if len(line) > 1:
            hash,dependency = line.split(" ", 1)
            dependency = dependency[2:-2]
            parts = dependency.split("),(")
            dep_list = []
            for part in parts:
            # (had,VBD),punct,(,,,)  (ROOT),ROOT,(had,VBD)
                gov, list = part.split("),", 1) # (had,VBD   punct,(,,,)
                dep_rel, dep = list.split(",(",1)  # punct ,,,)
                if ("," in gov) and ("," in dep) and ("-" not in gov) and ("-" not in dep):
                    if len(gov.split(","))==2 and  len(dep.split(",")) == 2:
                        word1 = gov[1:].strip().split(",")
                        word2 = dep[:-1].strip().split(",")
                        dep_list.append((word1,dep_rel.strip(),word2))
            dependency_list[hash] = dep_list
    return dependency_list


# (ROOT),ROOT,(price,NN) (price,NN),case,(for,IN))
    print("")
    file_name.close()
    return dependency_result


def double_propagation(dependency_list, opinion_seed, feature_seed):
    opinion_set = {}
    for k in opinion_seed:
        opinion_set[k] = OpinionWord(k.lower(), features_modified=set(), opinions_modified=set(),
                                       sentences_from=set(), extracting_rules=set())

    feature_set = {}
    for k in feature_seed:
        feature_set[k] = FeatureWord(k.lower(), features_modified=set(), sentences_from=set(), extracting_rules=set())

    old_opinion_size = len(opinion_set)
    old_feature_size = len(feature_set)
    i = 1
    while True:
        rule.rule_one_one(dependency_list,feature_set, opinion_set)
        rule.rule_one_two(dependency_list, feature_set, opinion_set)

        rule.rule_four_one(dependency_list, opinion_set)
        rule.rule_four_two(dependency_list, opinion_set)

        rule.rule_three_one(dependency_list, feature_set)
        rule.rule_three_two(dependency_list, feature_set)

        rule.rule_two_one(dependency_list, opinion_set, feature_set)
        rule.rule_two_two(dependency_list, opinion_set, feature_set)

        if (len(feature_set) - old_feature_size) < 1 and (len(opinion_set) - old_opinion_size) < 1:
            break

        old_feature_size = len(feature_set)
        old_opinion_size = len(opinion_set)
        i += 1
    return opinion_set, feature_set


def dep2pairs(in_file, out_file):
    # load sentiment seed words
    sentiment_file = open(Config.data_path + Config.sentiment_seed, "r")
    sentiment_seed = set(sentiment_file.read().split())
    feature_seed = set()

    in_file = open(in_file,"r")
    dependency_result = {}
    dependency_result = extract_depency(in_file)

    opinion_set, feature_set = double_propagation(
        dependency_result,
        sentiment_seed,
        feature_seed)
    out_file = open(out_file,"w")
    for k in opinion_set:
        if k.isalpha():
            for m in opinion_set[k].features_modified:
                if m.isalpha() and len(m) != 0:
                        n = feature_set[m].sentences_from & opinion_set[k].sentences_from
                        if len(n) != 0:
                            for l in n:
                                out_file.write(l + ' ')
                                out_file.write(k + ',')
                                out_file.write(m + '\n')



    in_file.close()
    out_file.close()

if __name__ == "__main__":

    # for file in Config.file_name:
    #     dep2pairs(Config.data_path + file + "_parsed.txt",Config.data_path + file + "_pairs.txt")
    # dep2pairs("Apex AD2600 Progressive-scan DVD player_parsed.txt", "Apex AD2600 Progressive-scan DVD player_pairs.txt")

    # file_name = open("Apex AD2600 Progressive-scan DVD player_parsed.txt")
    # dependency_result = extract_depency(file_name)
    dep2pairs("../source/ABSA-15_Laptops_Train_Data_parsed.txt", "../source/ABSA-15_Laptops_Train_Data_pairs.txt")
    dep2pairs("../source/ABSA-15_Restaurants_Train_Final_parsed.txt","../source/ABSA-15_Restaurants_Train_Final_pairs.txt")









