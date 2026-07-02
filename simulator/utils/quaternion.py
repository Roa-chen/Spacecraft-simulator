
import numpy as np

"""
Quaternions are considered with scalar part being in the first position, and as line matricies
"""

def normalize(q: np.ndarray)->float:
    np.linalg.norm(q)
    
def normalize_quaternions_state(state, state_indices):
    attitude = state[state_indices["ATTITUDE"]].reshape(-1, 4)
    attitude = np.einsum('ni,n->ni', attitude, 1/np.linalg.norm(attitude, axis=-1))
    state[state_indices["ATTITUDE"]] = attitude.flatten()

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
    


