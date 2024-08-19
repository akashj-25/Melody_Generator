import os
import json
import music21 as m21
import numpy as np
import tensorflow.keras as keras

KERN_DATASET_PATH = "deutschl/erk"
SAVE_DIR = "dataset"
SINGLE_FILE_DATASET = "file_dataset"
MAPPING_PATH = "mapping.json"
SEQUENCE_LENGTH = 64

# List of acceptable durations in quarter lengths (e.g., 0.25 for 16th note)
ACCEPTABLE_DURATIONS = [
    0.25,
    0.5,
    0.75,
    1.0,
    1.5,
    2,
    3,
    4
]


def load_songs_in_kern(dataset_path):
    """Loads all songs from the dataset in Kern format.

    :param dataset_path (str): Directory containing the dataset files.
    :return songs (list of m21 streams): List of loaded songs as music21 streams.
    """
    songs = []

    # Walk through all files in the dataset directory
    for path, subdirs, files in os.walk(dataset_path):
        for file in files:
            # Load only Kern files (.krn)
            if file.endswith(".krn"):
                song = m21.converter.parse(os.path.join(path, file))
                songs.append(song)
    return songs


def has_acceptable_durations(song, acceptable_durations):
    """Checks if all note/rest durations in the song are within the acceptable list.

    :param song (m21 stream): A music21 stream object representing the song.
    :param acceptable_durations (list): List of acceptable durations in quarter lengths.
    :return (bool): True if all durations are acceptable, False otherwise.
    """
    for note in song.flat.notesAndRests:
        if note.duration.quarterLength not in acceptable_durations:
            return False
    return True


def transpose(song):
    """Transposes a song to C major or A minor.

    :param song (m21 stream): A music21 stream object representing the song.
    :return transposed_song (m21 stream): The transposed song.
    """
    # Retrieve the key of the song
    parts = song.getElementsByClass(m21.stream.Part)
    measures_part0 = parts[0].getElementsByClass(m21.stream.Measure)
    key = measures_part0[0][4]

    # Analyze the key if it's not explicitly provided
    if not isinstance(key, m21.key.Key):
        key = song.analyze("key")

    # Determine the interval to transpose the song to C major or A minor
    if key.mode == "major":
        interval = m21.interval.Interval(key.tonic, m21.pitch.Pitch("C"))
    elif key.mode == "minor":
        interval = m21.interval.Interval(key.tonic, m21.pitch.Pitch("A"))

    # Transpose the song
    transposed_song = song.transpose(interval)
    return transposed_song


def encode_song(song, time_step=0.25):
    """Encodes a song into a time-series representation.

    :param song (m21 stream): A music21 stream object representing the song.
    :param time_step (float): Duration of each time step in quarter lengths.
    :return encoded_song (str): A string representing the encoded song.
    """
    encoded_song = []

    for event in song.flat.notesAndRests:
        # Convert notes to MIDI numbers and rests to 'r'
        if isinstance(event, m21.note.Note):
            symbol = event.pitch.midi
        elif isinstance(event, m21.note.Rest):
            symbol = "r"

        # Calculate the number of steps for the duration and encode the symbol
        steps = int(event.duration.quarterLength / time_step)
        for step in range(steps):
            if step == 0:
                encoded_song.append(symbol)
            else:
                encoded_song.append("_")

    # Convert the encoded list to a string
    encoded_song = " ".join(map(str, encoded_song))

    return encoded_song


def preprocess(dataset_path):
    """Preprocess the dataset by loading, filtering, transposing, encoding, and saving the songs.

    :param dataset_path (str): Directory containing the dataset files.
    """
    print("Loading songs...")
    songs = load_songs_in_kern(dataset_path)
    print(f"Loaded {len(songs)} songs.")

    for i, song in enumerate(songs):
        # Filter out songs with unacceptable durations
        if not has_acceptable_durations(song, ACCEPTABLE_DURATIONS):
            continue

        # Transpose the song and encode it
        song = transpose(song)
        encoded_song = encode_song(song)

        # Save the encoded song to a text file
        save_path = os.path.join(SAVE_DIR, str(i))
        with open(save_path, "w") as fp:
            fp.write(encoded_song)

        # Log progress every 10 songs
        if i % 10 == 0:
            print(f"Song {i} out of {len(songs)} processed")


def load(file_path):
    """Loads the contents of a file as a string.

    :param file_path (str): Path to the file.
    :return song (str): The contents of the file.
    """
    with open(file_path, "r") as fp:
        song = fp.read()
    return song


def create_single_file_dataset(dataset_path, file_dataset_path, sequence_length):
    """Combines all encoded songs into a single file with delimiters between them.

    :param dataset_path (str): Directory containing the encoded songs.
    :param file_dataset_path (str): Path to save the combined dataset file.
    :param sequence_length (int): Length of each sequence to be used for training.
    :return songs (str): A string containing all encoded songs with delimiters.
    """
    new_song_delimiter = "/ " * sequence_length
    songs = ""

    # Load each encoded song and concatenate them with delimiters
    for path, _, files in os.walk(dataset_path):
        for file in files:
            file_path = os.path.join(path, file)
            song = load(file_path)
            songs = songs + song + " " + new_song_delimiter

    # Remove trailing space and save the combined dataset
    songs = songs.rstrip()

    with open(file_dataset_path, "w") as fp:
        fp.write(songs)

    return songs


def create_mapping(songs, mapping_path):
    """Creates a mapping of symbols to integers and saves it as a JSON file.

    :param songs (str): A string containing all encoded songs.
    :param mapping_path (str): Path to save the mapping file.
    """
    mappings = {}

    # Identify unique symbols (vocabulary) in the songs
    songs = songs.split()
    vocabulary = list(set(songs))

    # Map each symbol to an integer
    for i, symbol in enumerate(vocabulary):
        mappings[symbol] = i

    # Save the mapping to a JSON file
    with open(mapping_path, "w") as fp:
        json.dump(mappings, fp, indent=4)


def convert_songs_to_int(songs):
    """Converts the encoded songs from symbols to integers based on the mapping.

    :param songs (str): A string containing all encoded songs.
    :return int_songs (list of int): A list of integers representing the songs.
    """
    int_songs = []

    # Load the mappings from the JSON file
    with open(MAPPING_PATH, "r") as fp:
        mappings = json.load(fp)

    # Convert each symbol in the songs to its corresponding integer
    songs = songs.split()
    for symbol in songs:
        int_songs.append(mappings[symbol])

    return int_songs


def generate_training_sequences(sequence_length):
    """Generates input and target sequences for model training.

    :param sequence_length (int): Length of each input sequence.
    :return inputs (ndarray): One-hot encoded input sequences.
    :return targets (ndarray): Target symbols for each input sequence.
    """
    # Load and convert songs to integers
    songs = load(SINGLE_FILE_DATASET)
    int_songs = convert_songs_to_int(songs)

    inputs = []
    targets = []

    # Create input and target sequences
    num_sequences = len(int_songs) - sequence_length
    for i in range(num_sequences):
        inputs.append(int_songs[i:i+sequence_length])
        targets.append(int_songs[i+sequence_length])

    # One-hot encode the input sequences
    vocabulary_size = len(set(int_songs))
    inputs = keras.utils.to_categorical(inputs, num_classes=vocabulary_size)
    targets = np.array(targets)

    print(f"There are {len(inputs)} sequences.")

    return inputs, targets


def main():
    """Main function to preprocess the dataset and generate training sequences."""
    preprocess(KERN_DATASET_PATH)
    songs = create_single_file_dataset(SAVE_DIR, SINGLE_FILE_DATASET, SEQUENCE_LENGTH)
    create_mapping(songs, MAPPING_PATH)
    # Uncomment the line below to generate training sequences
    # inputs, targets = generate_training_sequences(SEQUENCE_LENGTH)


if __name__ == "__main__":
    main()
