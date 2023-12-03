import tensorflow as tf
from intents.get_intents import IntentsManager
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, GlobalAveragePooling1D


class NLPModel:
    def __init__(self, intents_data_, model_path_, word_index_, epochs_=1000, verbose_=2):
        self.intents_data_ = intents_data_
        self.word_index_ = word_index_
        self.epochs_ = epochs_
        self.model_path_ = model_path_
        self.verbose_ = verbose_
        self.X_train, self.Y_train = self.get_training_data()

    def get_training_data(self):
        x_train = []
        y_train = []

        # Create training data
        for idx, intent in enumerate(self.intents_data_["intents"]):
            for pattern in intent["patterns"]:
                x_train.append([self.word_index_[word.lower()] for word in pattern.split()])
                y_train.append(idx)

        return x_train, y_train

    def build_model(self):
        try:
            # Convert X_train to a fixed size using padding
            max_length = max(len(pattern) for pattern in self.X_train)
            x_train = tf.keras.preprocessing.sequence.pad_sequences(self.X_train, maxlen=max_length)

            # Convert y_train to one-hot encoding
            y_train = tf.keras.utils.to_categorical(self.Y_train, num_classes=len(self.intents_data_["intents"]))

            # Build the model
            model = Sequential()
            model.add(Embedding(input_dim=len(self.word_index_) + 1, output_dim=32, input_length=max_length))
            model.add(GlobalAveragePooling1D())
            model.add(Dense(len(self.intents_data_["intents"]), activation="softmax"))

            # Compile the model
            model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

            # Train the model
            model.fit(x_train, y_train, epochs=self.epochs_, verbose=self.verbose_)

            # Save the model
            model.save(self.model_path_)

        except Exception as e:
            print("Error building and training the model:", str(e))

    def load_model(self):
        try:
            loaded_model = tf.keras.models.load_model(self.model_path_)
            return loaded_model
        except Exception as e:
            print("Error loading the model:", str(e))
            return None

    def get_patterns_length(self):
        max_length = max(len(pattern) for pattern in self.X_train)
        return max_length


if __name__ == '__main__':
    try:
        intents_manager = IntentsManager()
        _, word_idx = intents_manager.get_words_and_index()
        DEFAULT_MODEL_PATH = "model.h5"

        nlp_model = NLPModel(intents_manager.get_intents(), DEFAULT_MODEL_PATH, word_idx)
        nlp_model.build_model()

    except Exception as e:
        print("Error initializing and training the NLP model:", str(e))
