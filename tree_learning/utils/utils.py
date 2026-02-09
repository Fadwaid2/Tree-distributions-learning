import gc
import numpy as np

def refresh_dpt(old_dpt):
    new_dpt = {}
    for key, arr in old_dpt.items():
        if arr.size>1:  
            new_dpt[key] = arr[1:]  #remove the first element for low memory 
        else:
            new_dpt[key] = np.array([])  
    del old_dpt  
    gc.collect()  #empty old DPT's memory with garbage collector 
    return new_dpt

def sanitize_edges(edges):
    sanitized = []
    for edge in edges:
        if isinstance(edge, (list, np.ndarray)) and len(edge) == 2:
            sanitized.append(tuple(edge))
        elif isinstance(edge, tuple) and len(edge) == 2:
            sanitized.append(edge)
        else:
            raise ValueError(f"Invalid edge format: {edge}")
    return sanitized

