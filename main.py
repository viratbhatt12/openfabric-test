import random
import tensorflow as tf
from model.model import NLPModel
from intents.get_intents import IntentsManager
from openfabric_pysdk.loader import ConfigClass
from openfabric_pysdk.context import OpenfabricExecutionRay
from ontology_dc8f06af066e4a7880a5938933236037.simple_text import SimpleText

# Load intents data
intents_manager = IntentsManager()
intents_data = intents_manager.get_intents()
words, word_idx = intents_manager.get_words_and_index()

# Initialize NLP model
DEFAULT_SAVED_MODEL_PATH = "model/model.h5"
nlp_model = NLPModel(intents_manager.get_intents(), DEFAULT_SAVED_MODEL_PATH, word_idx)
max_length = nlp_model.get_patterns_length()
loaded_model = nlp_model.load_model()


############################################################
# Callback function called on update config
############################################################
def config(configuration: ConfigClass):
    # TODO Add code here
    pass


############################################################
# Callback function called on each execution pass
############################################################
def execute(request: SimpleText, ray: OpenfabricExecutionRay) -> SimpleText:
    """
    Callback function called on each execution pass.

    :param request: Input request.
    :param ray: Openfabric execution ray.
    :return: SimpleText response.
    """
    output = []
    try:
        for text in request.text:
            response = get_response(text)
            output.append(response)
    except Exception as e:
        # Handle errors gracefully and provide a meaningful response
        output.append("Apologies, there was an issue processing your request. Please try again later.")
        print("Error executing callback:", str(e))

    return SimpleText(dict(text=output))


def preprocess_input(user_input_, word_index_, max_length_):
    """
    Preprocess user input for model prediction.

    :param user_input_: User input text.
    :param word_index_: Word index dictionary.
    :param max_length_: Maximum length of input sequences.
    :return: Preprocessed input.
    """
    try:
        user_input = [word_index_.get(word.lower(), 0) for word in user_input_.split()]
        user_input = tf.keras.preprocessing.sequence.pad_sequences([user_input], maxlen=max_length_)
        return user_input
    except Exception as e:
        raise Exception("Error preprocessing input: {}".format(str(e)))


def get_response(user_input):
    """
    Get a response from the NLP model based on user input.

    :param user_input: User input text.
    :return: Generated response.
    """
    try:
        processed_input = preprocess_input(user_input, word_idx, max_length)
        prediction = loaded_model.predict(processed_input)
        intent_index = tf.argmax(prediction, axis=1).numpy()[0]
        confidence = prediction[0][intent_index]

        # If confidence is below a certain threshold, provide a generic response
        if confidence < 0.5:
            return "I'm not completely sure about that, but here's something relevant: " + random.choice(
                intents_data["intents"][intent_index]['responses'])

        response = random.choice(intents_data["intents"][intent_index]['responses'])
        return response
    except Exception as e:
        # Handle errors gracefully and provide a meaningful response
        print("Error getting response:", str(e))
        return "Apologies, there was an issue generating a response. Please try again later."
