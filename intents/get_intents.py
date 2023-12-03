import json

DEFAULT_INTENTS_PATH = "intents/intents.json"


class IntentsManager:
    def __init__(self):
        try:
            with open(DEFAULT_INTENTS_PATH, "r") as intents_file:
                intents = json.load(intents_file)
        except FileNotFoundError:
            print(f"Intents file is not found in {DEFAULT_INTENTS_PATH}")
            intents = {}

        self.__intents = intents
        self.patterns, self.responses = self.get_patterns_and_responses()

    def get_intents(self):
        try:
            # Get intents
            return self.__intents
        except Exception as e:
            print("Error retrieving intents:", str(e))
            return {}

    def get_patterns_and_responses(self):
        try:
            patterns = []
            responses = []

            # Extract patterns and responses from intents
            for intent in self.__intents["intents"]:
                patterns.extend(intent["patterns"])
                responses.extend(intent["responses"])

            return patterns, responses
        except Exception as e:
            print("Error extracting patterns and responses:", str(e))
            return [], []

    def get_words_and_index(self):
        try:
            # Create a set of unique words
            words = set([word.lower() for pattern in self.patterns for word in pattern.split()])

            # Create a dictionary to map words to integers
            word_index = {word: idx + 1 for idx, word in enumerate(sorted(list(words)))}
            return words, word_index
        except Exception as e:
            print("Error creating words and index:", str(e))
            return set(), {}
