from typing import Optional

import numpy as np
import operator as op

from functools import reduce


class HammingMessage:
    def __init__(self, message: np.ndarray):
        self._raw_message = message
        self.message: Optional[np.array] = None

    def check(self):
        return self.find_unset_parity_bit(self.message)

    @staticmethod
    def find_unset_parity_bit(bits: np.ndarray) -> int:
        return reduce(op.xor, [i for i, b in enumerate(bits) if b])

    def prepare(self):
        """

        :param message: ndarray of bits
        :return: message ready with hamming parity bits
        """
        message = self._raw_message

        unset_bit = self.find_unset_parity_bit(message)
        while unset_bit:
            message[unset_bit] = not message[unset_bit]
            unset_bit = self.find_unset_parity_bit(message)
        assert unset_bit == 0

        self.message = message

    def print(self):
        message_size = len(self._raw_message)
        assert np.math.log(message_size, 2) - int(np.math.log(message_size, 2)) == 0
        square_length = int(np.sqrt(message_size))
        print(random_message.reshape(square_length, square_length), end="\n"*2)


if __name__ == "__main__":
    message_size = 16
    assert np.math.log(message_size, 2) - int(np.math.log(message_size, 2)) == 0
    random_message = np.random.randint(0, 2, message_size)
    hamming = HammingMessage(random_message)
    hamming.print()
    hamming.prepare()
    corrupted_bit_offset = 10
    hamming.message[corrupted_bit_offset] = not hamming.message[corrupted_bit_offset]
    hamming.print()
    print(hamming.check())
