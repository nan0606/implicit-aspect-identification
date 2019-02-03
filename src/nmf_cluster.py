# -*- coding:utf-8 -*-
# _author_ = xu_qn
# -*- coding:utf-8 -*-
# _author_ = xu_qn
import src.feature_opinion_matrix as fom
import numpy as np
from nltk.corpus import wordnet as wn
from stop_words import get_stop_words
from nltk.tokenize import RegexpTokenizer
import math
import src.control_experiment_9 as ce9
import collections


en_stop = get_stop_words("en")
tokenizer = RegexpTokenizer(r'\w+')

def opinion_feature_matrix(feature_dict,opinion_dict,corpus):
    """统计opiion-feature矩阵，共现频率"""
    C = [[0.0 for i in range(len(feature_dict.keys()))]for j in range(len(opinion_dict))]
    for sent in corpus:
        for feature in feature_dict:
            for opinion in opinion_dict:
                if feature in sent and opinion in sent:
                    C[opinion_dict[opinion]][feature_dict[feature]] += 1
    return C

def gennerate_feature_matrix(feature_dict):
    """根据word net生成路径长度矩阵"""
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
    """根据是否是同义词反义词生成矩阵"""
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
    """得到拉普拉斯矩阵"""
    W = np.array(W)
    m = len(W[1])
    D = np.mat(np.zeros((m, m)))
    if (W.T == W).all():
        for i in range(m):
            D[i, i] = W[i].sum()
    L = D - W
    return D, L

def train0(X, W_A, W_O,D_A,D_O, k, iter_num, err, alpha_a, alpha_o, beta):
    """迭代生成两个矩阵"""
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

       # 更新U,V
       #  a = X * V
       #  b = U * V.T * V
        a = X * V + alpha_o * W_O * U
        b = U * V.T * V+ alpha_o * D_O * U
        for i in range(m):
            for j in range(k):
                if b[i,j] != 0:
                    U[i,j] = U[i,j] * a[i,j] / b[i,j]

        c = X.T * U
        d = V * U.T * U
        for i in range(n):
            for j in range(k):
                if d[i,j]  != 0:
                    V[i,j] = V[i,j] * c[i,j] / (d[i,j])

    return U,V

def train1(X, W_A, W_O,D_A,D_O, k, iter_num, err, alpha_a, alpha_o, beta):
    """迭代生成两个矩阵"""
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

       # 更新U,V
        a = X * V
        b = U * V.T * V
        for i in range(m):
            for j in range(k):
                if b[i,j] != 0:
                    U[i,j] = U[i,j] * a[i,j] / b[i,j]

        c = X.T * U
        d = V * U.T * U
        for i in range(n):
            for j in range(k):
                orth = 0
                for b in range(k):
                    if b != j:
                        orth += 2*beta * V[:,b].T * V[:,j] * V[i,b]
                if d[i,j] + orth != 0:
                    V[i,j] = V[i,j] * c[i,j] / (d[i,j] + orth)

    return U,V

def train2(X, W_A, W_O,D_A,D_O, k, iter_num, err, alpha_a, alpha_o, beta):
    """迭代生成两个矩阵"""
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

       # 更新U,V
        a = X * V
        b = U * V.T * V
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

    return U,V

def train3(X, W_A, W_O,D_A,D_O, k, iter_num, err, alpha_a, alpha_o, beta):
    """迭代生成两个矩阵"""
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

       # 更新U,V
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

    return U,V


def get_max_index(matrix):
    """得到矩阵中每一列最大的值"""
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

def main_test0(file, k, true_cluster_res):
    # k = 20
    iter_num = 50
    err = 0.5
    alpha_a, alpha_o = 0.2, 0.2
    beta = 0.2
    # file = "../data/Apex AD2600 Progressive-scan DVD player"
    file_pairs = file + "_pairs.txt"
    feature_dict, opinion_dict = ce9.read_feature_opinion_dict(file_pairs)

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
    U, V = train0(X, W_A, W_O, D_A, D_O, k, iter_num, err, alpha_a, alpha_o, beta)
    # print(U)

    # 对V的每一行，找到最大的列索引得到聚类结果[0,1,2,3] feature_dict{w:index} 变为[[簇1中的特征集合]]
    cluster_res = get_max_index(V)

    #  根据真实的聚类结果得到目标值
    from sklearn import metrics

    # true_cluster_res = [15, 0, 18, 14, 7, 17, 10, 17, 3, 4, 5, 5, 5, 11, 7, 11, 18, 14, 11, 1, 0, 3, 10, 15, 16, 2, 19,
    #                     6, 0, 9, 10, 0, 13, 4, 3, 7, 1, 7, 16, 12, 19, 17, 9, 4, 5, 3, 2, 7, 14, 13, 12, 1, 2, 6, 2, 3,
    #                     6, 16, 7, 18, 13, 13, 1, 6, 4, 8, 19]
    res_arry = np.array(cluster_res)
    true_res_arry = np.array(true_cluster_res)
    return metrics.adjusted_rand_score(res_arry, true_res_arry)

def main_test1(file, k, true_cluster_res):
    # k = 20
    iter_num = 50
    err = 0.5
    alpha_a, alpha_o = 0.2, 0.2
    beta = 0.2
    # file = "../data/Apex AD2600 Progressive-scan DVD player"
    file_pairs = file + "_pairs.txt"
    feature_dict, opinion_dict = ce9.read_feature_opinion_dict(file_pairs)

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
    U, V = train1(X, W_A, W_O, D_A, D_O, k, iter_num, err, alpha_a, alpha_o, beta)
    # print(U)

    # 对V的每一行，找到最大的列索引得到聚类结果[0,1,2,3] feature_dict{w:index} 变为[[簇1中的特征集合]]
    cluster_res = get_max_index(V)

    #  根据真实的聚类结果得到目标值
    from sklearn import metrics

    # true_cluster_res = [15, 0, 18, 14, 7, 17, 10, 17, 3, 4, 5, 5, 5, 11, 7, 11, 18, 14, 11, 1, 0, 3, 10, 15, 16, 2, 19,
    #                     6, 0, 9, 10, 0, 13, 4, 3, 7, 1, 7, 16, 12, 19, 17, 9, 4, 5, 3, 2, 7, 14, 13, 12, 1, 2, 6, 2, 3,
    #                     6, 16, 7, 18, 13, 13, 1, 6, 4, 8, 19]
    res_arry = np.array(cluster_res)
    true_res_arry = np.array(true_cluster_res)
    return metrics.adjusted_rand_score(res_arry, true_res_arry)

def main_test2(file, k, true_cluster_res):
    # k = 20
    iter_num = 50
    err = 0.5
    alpha_a, alpha_o = 0.2, 0.2
    beta = 0.2
    # file = "../data/Apex AD2600 Progressive-scan DVD player"
    file_pairs = file + "_pairs.txt"
    feature_dict, opinion_dict = ce9.read_feature_opinion_dict(file_pairs)

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
    U, V = train2(X, W_A, W_O, D_A, D_O, k, iter_num, err, alpha_a, alpha_o, beta)
    # print(U)

    # 对V的每一行，找到最大的列索引得到聚类结果[0,1,2,3] feature_dict{w:index} 变为[[簇1中的特征集合]]
    cluster_res = get_max_index(V)

    #  根据真实的聚类结果得到目标值
    from sklearn import metrics

    # true_cluster_res = [15, 0, 18, 14, 7, 17, 10, 17, 3, 4, 5, 5, 5, 11, 7, 11, 18, 14, 11, 1, 0, 3, 10, 15, 16, 2, 19,
    #                     6, 0, 9, 10, 0, 13, 4, 3, 7, 1, 7, 16, 12, 19, 17, 9, 4, 5, 3, 2, 7, 14, 13, 12, 1, 2, 6, 2, 3,
    #                     6, 16, 7, 18, 13, 13, 1, 6, 4, 8, 19]
    res_arry = np.array(cluster_res)
    true_res_arry = np.array(true_cluster_res)
    return metrics.adjusted_rand_score(res_arry, true_res_arry)

def main_test3(file, k, true_cluster_res):
    # k = 20
    iter_num = 50
    err = 0.5
    alpha_a, alpha_o = 0.2, 0.2
    beta = 0.2
    # file = "../data/Apex AD2600 Progressive-scan DVD player"
    file_pairs = file + "_pairs.txt"
    feature_dict, opinion_dict = ce9.read_feature_opinion_dict(file_pairs)

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
    U, V = train3(X, W_A, W_O, D_A, D_O, k, iter_num, err, alpha_a, alpha_o, beta)
    # print(U)

    # 对V的每一行，找到最大的列索引得到聚类结果[0,1,2,3] feature_dict{w:index} 变为[[簇1中的特征集合]]
    cluster_res = get_max_index(V)

    #  根据真实的聚类结果得到目标值
    from sklearn import metrics

    # true_cluster_res = [15, 0, 18, 14, 7, 17, 10, 17, 3, 4, 5, 5, 5, 11, 7, 11, 18, 14, 11, 1, 0, 3, 10, 15, 16, 2, 19,
    #                     6, 0, 9, 10, 0, 13, 4, 3, 7, 1, 7, 16, 12, 19, 17, 9, 4, 5, 3, 2, 7, 14, 13, 12, 1, 2, 6, 2, 3,
    #                     6, 16, 7, 18, 13, 13, 1, 6, 4, 8, 19]
    res_arry = np.array(cluster_res)
    true_res_arry = np.array(true_cluster_res)
    return metrics.adjusted_rand_score(res_arry, true_res_arry)
