import numpy as np
from pure_prng_package import pure_prng


def generate_heavyhash_matrix(matrix_seed: int) -> np.ndarray:
    generator = pure_prng(matrix_seed, 'xoshiro256++')
    result = np.arange(64 * 64).reshape((64, 64))

    while True:
        for i in range(64):
            for j in range(0, 64, 16):
                value = generator.source_random_number()
                for shift in range(16):
                    result[i][j + shift] = (value >> (4 * shift)) & 0xF

        # Emulate do...while behavior =)
        if (not is_4bit_precision(result)) or (not is_full_rank(result)):
            # start loop again
            continue
        # End generating matrix
        break

    return result


def is_4bit_precision(m: np.ndarray) -> bool:
    flat = m.flatten()
    return all(map(lambda i: 0 <= i <= 0xF, flat))


def is_full_rank(m: np.ndarray) -> bool:
    np.linalg.matrix_rank(m) == 64
