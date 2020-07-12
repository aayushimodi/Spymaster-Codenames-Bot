import nltk
from nltk.corpus import wordnet as wn
import numpy as np
import pickle as pk
import math
from sklearn.neighbors import NearestNeighbors

def main():
    #map = {}
    #with open('glove.6B.50d.txt', 'rb') as f:
    #    for l in f:
    #        line = l.decode().split()
    #        map[line[0]] = np.array(line[1:]).astype(np.float)
            #np.array takes in a list and returns a vector with the elements of the list
    with open('word2vec.pk', 'rb+') as f:
        map = pk.load(f)
    print("loaded")
    #nbrs = NearestNeighbors(n_neighbors=5, algorithm='ball_tree', metric = angularDistance).fit(np.array(list(map.values())))
    with open('knn.pk', 'rb+') as f:
        nbrs = pk.load(f)
    #with open('knn.pk', 'wb+') as f:
    #    pk.dump(nbrs, f)

    print("creating neighbors")
    #print(map['dog'].reshape(1,-1).shape)
    distances, indices = nbrs.kneighbors(map['boy'].reshape(1,-1))
    print(indices)
    for i in range(5):
        print(list(map.keys())[indices[0][i]])

def cosineSimilarity(a, b):
    result = a.dot(b)/(np.linalg.norm(a) * np.linalg.norm(b))
    if result > 1:
        return 1
    elif result < -1:
        return -1
    else:
        return result

def angularDistance(a, b):
    return math.acos(cosineSimilarity(a,b))/math.pi


main()