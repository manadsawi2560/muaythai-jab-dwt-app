import numpy as np
from typing import List, Tuple

def dtw(a: np.ndarray, b: np.ndarray, window: int = None) -> Tuple[float, List[Tuple[int,int]]]:
    """a: [Ta, D], b: [Tb, D] -> (cost, path) where path is list of (i,j)
    window: Sakoe-Chiba band radius (int frames) or None
    """
    Ta, Tb = a.shape[0], b.shape[0]
    D = np.full((Ta+1, Tb+1), np.inf, dtype=np.float64)
    D[0,0] = 0.0


    def dist(i,j):
        d = a[i] - b[j]
        return float(np.sqrt((d*d).sum()))


    for i in range(1, Ta+1):
        j_start = 1
        j_end = Tb+1
        if window is not None:
            j_start = max(1, i - window)
            j_end = min(Tb+1, i + window)
        for j in range(j_start, j_end):
            cost = dist(i-1, j-1)
            D[i, j] = cost + min(D[i-1, j ], # insertion
                            D[i, j-1], # deletion
                            D[i-1, j-1]) # match
    # backtrack
    i, j = Ta, Tb
    path = []
    while i>0 and j>0:
        path.append((i-1, j-1))
        steps = [(i-1, j), (i, j-1), (i-1, j-1)]
        prev = min(steps, key=lambda t: D[t])
        i, j = prev
    path.reverse()
    return float(D[Ta, Tb]), path