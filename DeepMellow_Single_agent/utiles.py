
import numpy as np 


def smoothing(x, level = 20):
    n = len(x)
    smoothed_x = []
    for i in range(n):
        if i < n - level +1: 
            smoothed_x.append(np.mean(x[i:i+level]))
        else:
            # print(i)
            smoothed_x.append(np.mean(x[i-level + 1:i + 1]))
            
    return np.array(smoothed_x)
                            
    