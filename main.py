import nltk
from nltk.corpus import wordnet as wn
import numpy as np

def main():
    map = {}
    with open('glove.6B.50d.txt', 'rb') as f:
        for l in f:
            line = l.decode().split()
            map[line[0]] = np.array(line[1:]).astype(np.float)
            #np.array takes in a list and returns a vector with the elements of the list
    f.close()



    girl = map['girl']
    boy = map['boy']
    king = map['king']
    queen = map['queen']
    antelope = map['antelope']
    elephant = map['elephant']

    quasiQueen = king - boy + girl
    print(cosineSimilarity(quasiQueen, queen)) # should be similar
    print(cosineSimilarity(boy, queen)) # should be less similar
    print(cosineSimilarity(antelope, girl)) # should not be similar
    print(cosineSimilarity(antelope, elephant))
    


    #   print(map['dog'])

def cosineSimilarity(a, b):
    return a.dot(b)/(np.linalg.norm(a) * np.linalg.norm(b))


main()