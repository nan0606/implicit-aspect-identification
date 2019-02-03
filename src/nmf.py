# -*- coding:utf-8 -*-
# _author_ = xu_qn
import src.feature_opinion_matrix as fom
import numpy as np
from nltk.corpus import wordnet as wn
from stop_words import get_stop_words
from nltk.tokenize import RegexpTokenizer
import math
import collections
import re


en_stop = get_stop_words("en")
tokenizer = RegexpTokenizer(r'\w+')

def read_feature_opinion_dict(file):
    """get aspect dictionary and opinion dictionary"""
    f = open(file,"r")
    feature_list = []
    opinion_list = []
    for line in f.readlines():
        hash, list = line.split(" ", 1)
        opinion, feature = list.split(",",1)
        if feature.strip() not in feature_list:
            feature_list.append(feature.strip())
        if opinion.strip() not in opinion_list:
            opinion_list.append(opinion.strip())
    feature_dict = dict(zip(feature_list,range(len(feature_list))))
    opinion_dict = dict(zip(opinion_list,range(len(opinion_list))))
    f.close()
    return feature_dict,opinion_dict


def opinion_feature_matrix(feature_dict,opinion_dict,corpus):
    """count co-occurrent, get opinion-feature matrix"""
    C = [[0.0 for i in range(len(feature_dict.keys()))]for j in range(len(opinion_dict))]
    for sent in corpus:
        for feature in feature_dict:
            for opinion in opinion_dict:
                if feature in sent and opinion in sent:
                    C[opinion_dict[opinion]][feature_dict[feature]] += 1
    return C

def gennerate_feature_matrix(feature_dict):
    """construct the aspect similarity matrix"""
    feature_similarity_matrix = [[0 for i in range(len(feature_dict.keys()))]for j in range(len(feature_dict.keys()))]
    for feature1,index1 in feature_dict.items():
        for feature2,index2 in feature_dict.items():
            if index1 == index2:
                feature_similarity_matrix[index1][index2] = 0
            elif len(wn.synsets(feature1)) == 0 or len(wn.synsets(feature2)) == 0:
                feature_similarity_matrix[index1][index2] = 0
            else:
                simi = wn.synsets(feature1)[0].path_similarity(wn.synsets(feature2)[0])
                if simi is None:
                    feature_similarity_matrix[index1][index2] = wn.synsets(feature2)[0].path_similarity(
                        wn.synsets(feature1)[0])
                    if feature_similarity_matrix[index1][index2] is None:
                        feature_similarity_matrix[index1][index2] =0
                else:
                    feature_similarity_matrix[index1][index2] = simi
    return feature_similarity_matrix

def generate_opinion_matrix(opinion_dict):
    """construct opinion affinity matrix"""
    opinion_similarity_matrix = [[0 for i in range(len(opinion_dict.keys()))] for j in range(len(opinion_dict.keys()))]
    for opinion1, index1 in opinion_dict.items():
        synonyms = set()
        antonyms = set()
        for syn in wn.synsets(opinion1):
            for l in syn.lemmas():
                synonyms.add(l.name())
                if l.antonyms():
                    antonyms.add(l.antonyms()[0].name())
        for opinion2, index2 in opinion_dict.items():
            if opinion2 in synonyms or opinion2 in antonyms:
                opinion_similarity_matrix[index1][index2] = 0
            else:
                opinion_similarity_matrix[index1][index2] = 1

    for i in range(len(opinion_similarity_matrix)):
        for j in range(len(opinion_similarity_matrix)):
            if opinion_similarity_matrix[i][j] != opinion_similarity_matrix[j][i]:
                opinion_similarity_matrix[i][j] = 1
                opinion_similarity_matrix[j][i] = 1

    return opinion_similarity_matrix

def get_Laplace_matrix(W):
    """get Laplacian matrix"""
    W = np.array(W)
    m = len(W[1])
    D = np.mat(np.zeros((m, m)))
    if (W.T == W).all():
        for i in range(m):
            D[i, i] = W[i].sum()
    L = D - W
    return D, L

def train(X, W_A, W_O,D_A,D_O, k, iter_num, err, alpha_a, alpha_o, beta):
    """NMF get V and U"""
    m, n = np.shape(X)
    X = np.mat(X)
    W_A, W_O = np.mat(W_A), np.mat(W_O)
    U, V = np.mat(np.random.random((m, k))), np.mat(np.random.random((n, k)))

    for x in range(iter_num):
        X_pre = U * V.T
        E = X - X_pre
        e = 0.0
        for i in range(m):
            for j in range(n):
                e += E[i,j] * E[i,j]
        if e < err:
            break

       # update
       #  a = X * V
       #  b = U * V.T * V
        a = X * V + alpha_o * W_O * U
        b = U * V.T * V+ alpha_o * D_O * U
        for i in range(m):
            for j in range(k):
                if b[i,j] != 0:
                    U[i,j] = U[i,j] * a[i,j] / b[i,j]

        # c = X.T * U
        # d = V * U.T * U
        c = X.T * U + alpha_a * W_A * V
        d = V * U.T * U + alpha_a * D_A * V
        for i in range(n):
            for j in range(k):
                orth = 0
                for b in range(k):
                    if b != j:
                        orth += 2*beta * V[:,b].T * V[:,j] * V[i,b]
                if d[i,j] + orth != 0:
                    V[i,j] = V[i,j] * c[i,j] / (d[i,j] + orth)
    V_std = V.std(axis=1)  # normalization
    V = V / V_std
    return U,V



def feature_clusters(feature_dict,V):
    """cluster features"""

    def get_max_index(matrix):
        m, n = np.shape(matrix)
        res_index = []
        for i in range(m):
            max = -0.1
            max_index = -1
            for j in range(n):
                if matrix[i,j] > max:
                    max = matrix[i,j]
                    max_index = j
            res_index.append(max_index)
        return res_index

    cluster_res = get_max_index(V)


    # from sklearn import metrics
    # true_cluster_res = [15,0,18,14,7,17,10,17,3,4,5,5,5,11,7,11,18,14,11,1,0,3,10,15,16,2,19,6,0,9,10,0,13,4,3,7,1,7,16,12,19,17,9,4,5,3,2,7,14,13,12,1,2,6,2,3,6,16,7,18,13,13,1,6,4,8,19]
    # res_arry = np.array(cluster_res)
    # true_res_arry = np.array(true_cluster_res)
    # print(metrics.adjusted_rand_score(res_arry,true_res_arry))


    _, n = np.shape(V)
    cluster_feature = [[] for i in range(n)]
    for i in range(len(cluster_res)):
        cluster_feature[cluster_res[i]].append(list(feature_dict.keys())[i])

    return cluster_feature


def find_review(file, hash_code):
    """
    :param file: XXX_text.txt, hash text
    :param hash_code:
    :return:
    """
    f = open(file, "r")
    for line in f.readlines():
        hash, text = line.split(" ",1)
        if hash == hash_code:
            return text.strip()
            break

def tag_reviews(feature_dict, file_pair, file_text):
    """
    @:param:pairs
    @:param:text
    @:return:{feature:{sentence1,sentence2}}
    """
    tag_dict = {}
    for i in feature_dict.keys():
        tag_dict[i] = []
    f = open(file_pair, "r")
    for line in f.readlines():
        hash_code, list = line.split(" ", 1)
        _,feature = list.split(",")
        sentence = find_review(file_text, hash_code)
        tag_dict[feature.strip()].append(sentence)
    f.close()
    return tag_dict

def cluster_review(tag_dict,cluster_feature):
    """
    :param tag_dict:
    :param cluster_feature:
    :return: review clusters
    """
    cluster_text = [[] for i in range(len(cluster_feature))]
    for feature,text in tag_dict.items():
        for i in range(len(cluster_feature)):
            if feature in cluster_feature[i]:
                cluster_text[i].extend(text)
    return cluster_text

def generate_vocabulary(corpus):
    """:return {w:index}"""
    texts = []
    for sent in corpus:
        token = tokenizer.tokenize(sent)
        for i in token:
            if i not in en_stop:
                texts.append(i)
    vocab = list(set(texts))
    vocab_dict = dict(zip(vocab,range(len(vocab))))
    return vocab_dict

def generate_centroid(vocab_dict,cluster_text,cluster_feature):
    """

    :param vocab_dict:
    :param cluster_text:
    :param cluster_feature:
    :return:the centroid vector of every cluster
    """
    centroid_list = [[0 for i in range(len(vocab_dict.keys()))]for j in range(len(cluster_feature))]
    # 1.count the number of categories whose review sets contains word
    word_cluster_list = [0 for i in range(len(vocab_dict.keys()))]
    for word,index in vocab_dict.items():
        for sents in cluster_text:
            text = [str1.split(" ") for str1 in sents]
            text_temp = [w for s in text for w in s]
            if word in text_temp:
                word_cluster_list[index] += 1
    # 2.count the word frequency
    for k in range(len(cluster_feature)):
        num_feature = len(cluster_feature[k])
        text = [str1.split(" ") for str1 in cluster_text[k]]
        text_temp = [w for s in text for w in s]
        if len(text_temp) == 0:
            for word, index in vocab_dict.items():
                centroid_list[k][index] = 0
        else:
            m = collections.Counter(text_temp)
            for word,index in vocab_dict.items():
                if word_cluster_list[index] == 0:
                    centroid_list[k][index] = 0
                else:
                    centroid_list[k][index] = m[word] * math.log(num_feature/word_cluster_list[index])

    return centroid_list

def vector_cosine(vector1, vector2):
    """compute the cosine similarity between two vectors"""
    dot_product = 0.0
    normA = 0.0
    normB = 0.0
    for a, b in zip(vector1, vector2):
        dot_product += a * b
        normA += a ** 2
        normB += b ** 2
    if normA == 0.0 or normB == 0.0:
        return 0.0
    else:
        return round(dot_product / ((normA**0.5)*(normB**0.5)) * 100, 2)

def predict_sentence(sentence,centroid_list,vocab_dict):
    """predict the cluster which the test sentence belong"""
    token = sentence.split(" ")
    sent_vector = [0 for i in range(len(vocab_dict.keys()))]
    for word,index in vocab_dict.items():
        if word in sentence:
            sent_vector[index] += 1 / len(token)
    max_index = -1
    max = float("-inf")
    for i in range(len(centroid_list)):
        num = vector_cosine(sent_vector,centroid_list[i])
        if num > max:
            max = num
            max_index = i
    return max_index

def rank_feature(cluster_feature,k, corpus, feature_dict, V, alpha = 0.5, threshold = 10):
    """
    rank aspect words of the cluster
    :param cluster_feature:
    :param k:
    :param corpus:
    :param V:
    :return: ranked feature_list [f1,f2]
    """

    feature_scores = {}
    for feature in cluster_feature[k]:
        feature_scores[feature] = 0

    corpus_temp = [sentence.split(" ") for sentence in corpus]
    corpus = [w for s in corpus_temp for w in s]
    m = collections.Counter(corpus)

    for feature in cluster_feature[k]:
        feature_scores[feature] = (1-alpha) * m[feature] + alpha * V[feature_dict[feature],k]

    feature_rank_list = sorted(feature_scores.items(), key=lambda item: item[1], reverse=True)
    if len(feature_rank_list) == 0:
        return  []
    elif feature_rank_list[0][1] < threshold:
        return []
    else:
        return feature_rank_list

def main_test(file, k):

    # k = 15
    iter_num = 50
    err = 0.5
    alpha_a, alpha_o, beta = 0.1,0.6,0.4
    # file = "../data/Apex AD2600 Progressive-scan DVD player"
    file_pairs = file + "_pairs.txt"
    feature_dict, opinion_dict = read_feature_opinion_dict(file_pairs)

    file_text = file + "_text.txt"
    corpus = []
    f = open(file_text, "r")
    for line in f.readlines():
        _, text = line.split(" ", 1)
        corpus.append(text.strip())
    f.close()

    X = opinion_feature_matrix(feature_dict, opinion_dict, corpus)

    W_A = gennerate_feature_matrix(feature_dict)
    W_O = generate_opinion_matrix(opinion_dict)
    D_A,_ = get_Laplace_matrix(W_A)
    D_O,_ = get_Laplace_matrix(W_O)
    U, V = train(X, W_A, W_O, D_A, D_O, k, iter_num, err, alpha_a, alpha_o, beta)
    # print(U)
    print(V)
    cluster_feature = feature_clusters(feature_dict, V)
    print(cluster_feature)
    # print(cluster_feature)
    tag_dict = tag_reviews(feature_dict, file_pairs, file_text)
    cluster_text = cluster_review(tag_dict, cluster_feature)
    vocab_dict = generate_vocabulary(corpus)
    centroid_list = generate_centroid(vocab_dict, cluster_text, cluster_feature)
    # # sentence = "it 's very sleek looking with a very good front panel button layout , and it has a great feature set"
    # # max_index = predict_sentence(sentence,centroid_list,vocab_dict)
    # # feature_rank_list = rank_feature(cluster_feature, max_index, corpus, feature_dict, V)
    # # print(feature_rank_list)
    #

    true_positive = 0
    false_positive = 0
    true_negative = 0
    false_negative = 0
    accuracy = 0  # predict accuracy
    sum_sentence = 0

    file_test = open(file + "_test.txt", "r")
    for line in file_test.readlines():
        _, list = line.split(" ", 1)
        feature, _, sentence = list.split("],")
        tag_features = re.split(",|#", feature[1:])
        max_index = predict_sentence(sentence, centroid_list, vocab_dict)
        res_list = [item[0] for item in rank_feature(cluster_feature, max_index, corpus, feature_dict, V, threshold= 4)]

        if tag_features != [""] and res_list != []:
            true_positive += 1
            sum_sentence += 1
            for tag_f in tag_features:
                if tag_f == res_list[0]:
                    accuracy += 1
        elif tag_features == [""] and res_list != []:
            false_positive += 1
        elif tag_features != [""] and res_list == []:
            false_negative += 1
        else:
            true_negative += 1

    return true_positive / (true_positive + false_positive), true_positive / (true_positive + false_negative), accuracy / sum_sentence

if __name__ == "__main__":
    for i in range(15):
        result = main_test("../data/Apex AD2600 Progressive-scan DVD player",15)
        print(result[0],result[1],result[2])
