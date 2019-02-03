# -*- coding:utf-8 -*-
# _author_ = xu_qn
from src.config import Config

def extract_test(file):
    """extract sentence with implicit aspect"""
    file_id = file + "_id.txt"
    file_pairs = file + "_pairs.txt"
    file_test = file + "_test.txt"

    train_list = []
    f_pairs = open(file_pairs, "r")
    for line in f_pairs.readlines():
        hash_code,_ = line.split(" ", 1)
        if hash_code not in train_list:
            train_list.append(hash_code)
    f_pairs.close()

    f_id = open(file_id, "r")
    f_test = open(file_test, "w", encoding="utf-8")

    for line in f_id.readlines():
        hash_code,_ = line.split(" ", 1)
        if hash_code not in train_list:
            f_test.write(line)

    f_id.close()
    f_test.close()

if __name__ == "__main__":
    for file in Config.file_name:
        extract_test("../source/ABSA-15_Restaurants_Train_Final")
