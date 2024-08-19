import json
import numpy as np
import tensorflow.keras as keras
import music21 as m21
from preprocess import SEQUENCE_LENGTH, MAPPING_PATH

class MelodyGenerator:
    """Generates melodies using a pre-trained LSTM model."""

    def __init__(self, model_path="model.h5"):
        # Load the trained LSTM model and note mappings
        self.model_path = model_path
        self.model = keras.models.load_model(model_path)

        with open(MAPPING_PATH, "r") as fp:
            self._mappings = json.load(fp)

        self._start_symbols = ["/"] * SEQUENCE_LENGTH  # Padding for initial sequence


    def generate_melody(self, seed, num_steps, max_sequence_length, temperature):
        """Generates a melody sequence based on a seed and returns it as a list of symbols.

        :param seed (str): The starting sequence for the melody.
        :param num_steps (int): Number of steps to generate after the seed.
        :param max_sequence_length (int): Maximum sequence length considered for model input.
        :param temperature (float): Controls randomness in the output (higher values = more randomness).

        :return melody (list of str): Generated sequence of melody symbols.
        """

        # Prepare the seed with start symbols and convert to integer representation
        seed = seed.split()
        melody = seed
        seed = self._start_symbols + seed
        seed = [self._mappings[symbol] for symbol in seed]

        for _ in range(num_steps):

            # Trim seed to fit the maximum input length for the model
            seed = seed[-max_sequence_length:]

            # One-hot encode the seed sequence for model input
            onehot_seed = keras.utils.to_categorical(seed, num_classes=len(self._mappings))
            onehot_seed = onehot_seed[np.newaxis, ...]

            # Predict the next symbol based on the current seed
            probabilities = self.model.predict(onehot_seed)[0]

            # Sample the next symbol using temperature to adjust randomness
            output_int = self._sample_with_temperature(probabilities, temperature)

            # Add the predicted symbol to the seed and melody
            seed.append(output_int)
            output_symbol = [k for k, v in self._mappings.items() if v == output_int][0]

            # Stop if the end symbol is reached
            if output_symbol == "/":
                break

            melody.append(output_symbol)

        return melody


    def _sample_with_temperature(self, probabilites, temperature):
        """Selects an index from a probability distribution after applying temperature scaling.

        :param predictions (nd.array): Model output probabilities for each possible symbol.
        :param temperature (float): Value to control randomness; lower values lead to more predictable outputs.

        :return index (int): The chosen index representing the next symbol.
        """
        predictions = np.log(probabilites) / temperature
        probabilites = np.exp(predictions) / np.sum(np.exp(predictions))

        # Sample an index based on the adjusted probabilities
        choices = range(len(probabilites))
        index = np.random.choice(choices, p=probabilites)

        return index


    def save_melody(self, melody, step_duration=0.25, format="midi", file_name="mel.mid"):
        """Converts a melody list into a MIDI file.

        :param melody (list of str): The generated melody as a list of symbols.
        :param step_duration (float): Duration of each step in quarter lengths.
        :param file_name (str): Name of the file to save the melody.
        """

        # Initialize a music21 stream to hold the notes/rests
        stream = m21.stream.Stream()

        start_symbol = None
        step_counter = 1

        # Convert each symbol in the melody to a music21 note/rest
        for i, symbol in enumerate(melody):

            if symbol != "_" or i + 1 == len(melody):

                if start_symbol is not None:
                    # Calculate the duration for the current note/rest
                    quarter_length_duration = step_duration * step_counter

                    # Create a Rest or Note object based on the symbol
                    if start_symbol == "r":
                        m21_event = m21.note.Rest(quarterLength=quarter_length_duration)
                    else:
                        m21_event = m21.note.Note(int(start_symbol), quarterLength=quarter_length_duration)

                    stream.append(m21_event)
                    step_counter = 1

                start_symbol = symbol

            else:
                # Increment duration if the symbol is a prolongation "_"
                step_counter += 1

        # Save the stream as a MIDI file
        stream.write(format, file_name)


if __name__ == "__main__":
    mg = MelodyGenerator()
    seed = "67 _ 67 _ 67 _ _ 65 64 _ 64 _ 64 _ _"
    seed2 = "67 _ _ _ _ _ 65 _ 64 _ 62 _ 60 _ _ _"
    melody = mg.generate_melody(seed, 500, SEQUENCE_LENGTH, 0.3)
    print(melody)
    mg.save_melody(melody)
