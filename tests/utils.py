import numpy as np
import hashlib

def compute_checksum(input_string):
    sha256 = hashlib.sha256()
    sha256.update(input_string.encode())
    return sha256.hexdigest()


def array_to_ascii_art(array, ascii_chars=".*:-=+#%@"):
    """
    Convert a 2D array to ASCII art.

    Parameters:
    - array: 2D numpy array
        The array to be converted into ASCII art.
    - ascii_chars: str, optional
        A string of characters used for ASCII art in increasing order of intensity.

    Returns:
    - str
        A string representing the ASCII art of the input array.
    """
    # Normalize the array
    normalized_array = (array - array.min()) / (array.max() - array.min() + 1e-8)

    # Map values to characters
    ascii_art = ""
    for row in normalized_array:
        ascii_art += ''.join([ascii_chars[int(val * (len(ascii_chars) - 1))] for val in row]) + "\n"

    return ascii_art


def test():
    result = array_to_ascii_art(np.array([[0,1,2,3,4,5,6,7]]).astype(int))[:-1]
    cs = compute_checksum('.*:-=+#%')
    assert cs == compute_checksum((result))

