import nltk
#nltk.download('wordnet')
from nltk.corpus import wordnet as wn

def main():
    print(wn.synsets('sofa')[0].definition())

main()