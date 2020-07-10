import nltk
from nltk.corpus import wordnet as wn

def main():
    map = {}
    with open('glove.6B.50d.txt', 'rb') as f:
        for l in f:
            line = l.decode().split()
            map[line[0]] = line[1:]

    f.close()

    print(map['dog'])

main()