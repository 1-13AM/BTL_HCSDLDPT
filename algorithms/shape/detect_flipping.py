import numpy as np


def remap_cch(cch_vector, mapping):
    remapped = [0] * 8
    for i in range(8):
        remapped[i] = cch_vector[mapping[i]]
    return remapped


def similarity_score(vec1, vec2):
    vec1, vec2 = np.array(vec1), np.array(vec2)
    dot = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    if norm1 == 0 or norm2 == 0:
        return 0
    return dot / (norm1 * norm2)


horizontal_map = [0, 7, 6, 5, 4, 3, 2, 1]
vertical_map = [4, 3, 2, 1, 0, 7, 6, 5]


def detect_flipping(vector1, vector2):
    hflip = remap_cch(vector2, horizontal_map)
    vflip = remap_cch(vector2, vertical_map)

    score_orig = similarity_score(vector1, vector2)
    score_h = similarity_score(vector1, hflip)
    score_v = similarity_score(vector1, vflip)

    result = {
        'original': score_orig,
        'horizontal': score_h,
        'vertical': score_v,
        'type': 'original'
    }

    if score_h > max(score_orig, score_v):
        result['type'] = 'horizontal'
    elif score_v > max(score_orig, score_h):
        result['type'] = 'vertical'

    return result
