import numpy as np
import math

class ShapeFunction:
    def evaluate(self, xi: float, eta: float):
        raise NotImplementedError("ShapeFunction.evaluate must be implemented by subclass.")

class Quadratic8ShapeFunction(ShapeFunction):
    def __init__(self):
        self.nodes_count = 4

    def evaluate(self, xi: float, eta: float):
        N = np.array([
            0.25 * (1 - xi) * (1 - eta),  
            0.25 * (1 + xi) * (1 - eta),  
            0.25 * (1 + xi) * (1 + eta),  
            0.25 * (1 - xi) * (1 + eta)   
        ])

        dN_dxi = np.array([
            -0.25 * (1 - eta),  
            0.25 * (1 - eta),  
            0.25 * (1 + eta),  
            -0.25 * (1 + eta)   
        ])

        dN_deta = np.array([
            -0.25 * (1 - xi),  
            -0.25 * (1 + xi),  
            0.25 * (1 + xi),  
            0.25 * (1 - xi)   
        ])

        return (N, dN_dxi, dN_deta)
