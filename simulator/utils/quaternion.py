
import numpy as np

"""
Quaternions are considered with scalar part being in the first position, and as line matricies
"""

def normalize(q: np.ndarray)->float:
    np.linalg.norm(q)

def multiply(q1, q2):
    return np.array([
        q1[0]*q2[0] - q1[1]*q2[1] - q1[2]*q2[2] - q1[3]*q2[3],
        q1[0]*q2[1] + q1[1]*q2[2] + q1[2]*q2[3] - q1[3]*q2[2],
        q1[0]*q2[2] - q1[1]*q2[3] + q1[2]*q2[0] + q1[3]*q2[1],
        q1[0]*q2[3] + q1[1]*q2[0] - q1[2]*q2[1] + q1[3]*q2[0],
    ])
    
def conjugate(q):
    return np.array([
        q[0],
        - q[1],
        - q[2],
        - q[3],
    ])
    


