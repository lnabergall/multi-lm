import numpy as np


def vectorize(sequences, max_sequence_length=None):
    """
    Args:
        inputs:
            list of sentences (integer lists)
        max_sequence_length:
            integer specifying how large should `max_time` dimension be.
            If None, maximum sequence length would be used
    
    Outputs:
        inputs_time_major:
            input sentences transformed into time-major matrix 
            (shape [max_time, batch_size]) padded with 0s
        sequence_lengths:
            batch-sized list of integers specifying amount of active 
            time steps in each input sequence
    """
    
    sequence_lengths = [len(sequence) for sequence in sequences]
    
    if max_sequence_length is None:
        max_sequence_length = max(sequence_lengths)
    
    inputs_batch_major = np.zeros(
        shape=[len(sequences), max_sequence_length], dtype=np.int8)
    
    for i, sequence in enumerate(sequences):
        for j, element in enumerate(sequence):
            inputs_batch_major[i, j] = element

    # [batch_size, max_time] -> [max_time, batch_size]
    # inputs_time_major = inputs_batch_major.swapaxes(0, 1)

    return inputs_batch_major, sequence_lengths


def random_sequences(length_from, length_to,
                     vocab_lower, vocab_upper,
                     batch_size, batches):
    """ 
    Returns a list of batches of random integer sequences,
    sequence length in [length_from, length_to],
    vocabulary in [vocab_lower, vocab_upper]
    """
    if length_from > length_to:
        raise ValueError('length_from > length_to')
    if vocab_lower > vocab_upper:
        raise ValueError('vocab_lower > vocab_upper')

    def random_length():
        if length_from == length_to:
            return length_from
        return np.random.randint(length_from, length_to + 1)
    
    sequences = []
    for i in range(batches):
        sequences.extend([np.random.randint(low=vocab_lower,
                                            high=vocab_upper,
                                            size=random_length()).tolist()
                          for _ in range(batch_size)])
    return sequences