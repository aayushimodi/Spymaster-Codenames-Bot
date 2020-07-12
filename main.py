import numpy as np
import pickle as pk
import math
import os
from sklearn.neighbors import NearestNeighbors


def main():
    map = loadWord2Vec()
    nbrs = loadNeighbors(map)
    words = loadWords()

    #print(map['dog'].reshape(1,-1).shape)
    distances, indices = nbrs.kneighbors(map['boy'].reshape(1,-1))
    print(indices)

    for i in range(5):
        print(list(map.keys())[indices[0][i]])


def loadWord2Vec():
    if os.path.isfile('word2vec.pk'):
        with open('word2vec.pk', 'rb+') as f:
            map = pk.load(f)
        print('word2vec loaded')
    else:
        map = {}
        with open('glove.6B.50d.txt', 'rb') as f:
            for l in f:
                line = l.decode().split()
                map[line[0]] = np.array(line[1:]).astype(np.float)
                # np.array takes in a list and returns a vector with the elements of the list

        with open('word2vec.pk', 'wb+') as f:
            pk.dump(map, f)
        print('word2vec created')

    return map


def loadNeighbors(map):
    if os.path.isfile('knn.pk'):
        with open('knn.pk', 'rb+') as f:
            nbrs = pk.load(f)
        print('neighbors loaded')
    else:
        nbrs = NearestNeighbors(n_neighbors=5, algorithm='ball_tree', metric=angularDistance).fit(
            np.array(list(map.values())))

        with open('knn.pk', 'wb+') as f:
            pk.dump(nbrs, f)
        print('neighbors created')

    return nbrs

def loadWords():
    # r = red, b = blue, n = neutral, x = death word
    words = {}
    with open('words.txt', 'r') as f:
        for l in f:
            line = l.split()
            words[line[0]] = line[1]
    return words

def cosineSimilarity(a, b):
    result = a.dot(b) / (np.linalg.norm(a) * np.linalg.norm(b))
    if result > 1:
        return 1
    elif result < -1:
        return -1
    else:
        return result


def angularDistance(a, b):
    return math.acos(cosineSimilarity(a, b)) / math.pi


main()
