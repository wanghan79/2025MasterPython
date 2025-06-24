from collections import defaultdict
from math import sqrt

def cosine_similarity(vec1, vec2):
    common = set(vec1.keys()) & set(vec2.keys())
    if not common:
        return 0
    num = sum(vec1[i] * vec2[i] for i in common)
    denom1 = sqrt(sum([v**2 for v in vec1.values()]))
    denom2 = sqrt(sum([v**2 for v in vec2.values()]))
    return num / (denom1 * denom2 + 1e-10)