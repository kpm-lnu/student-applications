import numpy as np
from scipy.interpolate import griddata

class AitkenOrder: 
    
    @staticmethod
    def aitken_order_interpolated(vertices1: np.ndarray, u1: np.ndarray, vertices2: np.ndarray, u2: np.ndarray, M2: np.ndarray, K2: np.ndarray, vertices3, u3: np.ndarray, M3: np.ndarray, K3: np.ndarray):
        u1_interp = griddata(vertices1, u1, vertices2, method='linear')
        u2_interp = griddata(vertices2, u2, vertices3, method='linear')

        delta_u2 = u2 - u1_interp
        delta_u3 = u3 - u2_interp

        L2_num = delta_u2.T @ M2 @ delta_u2
        L2_den = delta_u3.T @ M3 @ delta_u3

        H1_num = L2_num + delta_u2.T @ K2 @ delta_u2
        H1_den = L2_den + delta_u3.T @ K3 @ delta_u3

        p_L2 = np.log2(np.sqrt(L2_num / L2_den))
        p_H1 = np.log2(np.sqrt(H1_num / H1_den))

        return p_L2, p_H1