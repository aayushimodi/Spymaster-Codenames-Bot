import numpy as np

class GameboardGenerator:
    def __init__(self, word_to_index, nouns):
        self.word_to_index = word_to_index
        self.nouns = nouns

    def generateRandomBoard(self):
        sample = np.random.choice(len(self.nouns), size=25, replace=False)

        guessed = np.random.randint(0, high=2, size=25)
        return np.concatenate((sample, guessed))
 