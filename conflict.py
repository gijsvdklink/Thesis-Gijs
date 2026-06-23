import numpy as np

def safe_to_return(position_i, speed_i, position_j, speed_j):

    r = np.array(position_i) - np.array(position_j)
    v = np.array(speed_i) - np.array(speed_j)

    magnitude_v = np.linalg.norm(v)

    t_cpa = -(r @ v)/ magnitude_v**2

    vector_d_cpa = r + v * t_cpa

    absolute_d_cpa = np.linalg.norm(vector_d_cpa)

    return t_cpa, vector_d_cpa, absolute_d_cpa


pos_i = [2, 8]
v_i = [2,0]
pos_j = [8, 7]
v_j = [-1,-2]

t_cpa, vector_d_cpa, absolute_d_cpa = safe_to_return(pos_i, v_i, pos_j, v_j)
print(f"T_CPA = {t_cpa:.2f} seconds, d_cpa = {absolute_d_cpa:.2f} meter")