import numpy as np
import math

class ShapeFunction:
    def evaluate(self, xi: float, eta: float):
        raise NotImplementedError("ShapeFunction.evaluate must be implemented by subclass.")

class LinearQuadrilateralShapeFunction(ShapeFunction):
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

class Quadratic8ShapeFunction(ShapeFunction):
    def __init__(self):
        self.nodes_count = 8

    def evaluate(self, xi: float, eta: float):
        N = np.array([
            0.25 * (1 - xi) * (1 - eta) * (-xi - eta - 1),
            0.5 * (1 - math.pow(xi, 2)) * (1 - eta),
            0.25 * (1 + xi) * (1 - eta) * (xi - eta - 1),
            0.5 * (1 + xi) * (1 - math.pow(eta, 2)),
            0.25 * (1 + xi) * (1 + eta) * (xi + eta - 1),
            0.5 * (1 - math.pow(xi, 2)) * (1 + eta),
            0.25 * (1 - xi) * (1 + eta) * (-xi + eta - 1),
            0.5 * (1 - xi) * (1 - math.pow(eta, 2))
        ])

        dN_dxi = np.array([
            0.25 * (-(1 - eta) * (-xi - eta - 1) - (1 - xi) * (1 - eta)),  
            -xi * (1 - eta),                                              
            0.25 * ((1 - eta) * (xi - eta - 1) + (1 + xi) * (1 - eta)),   
            0.5 * (1 - eta**2),                                           
            0.25 * ((1 + eta) * (xi + eta - 1) + (1 + xi) * (1 + eta)),   
            -xi * (1 + eta),                                              
            0.25 * (-(1 + eta) * (-xi + eta - 1) - (1 - xi) * (1 + eta)), 
            -0.5 * (1 - eta**2)                                           
        ])
    
        dN_deta = np.array([
            0.25 * (-(1 - xi) * (-xi - eta - 1) - (1 - xi) * (1 - eta)),  
            -0.5 * (1 - xi**2),                                           
            0.25 * (-(1 + xi) * (xi - eta - 1) - (1 + xi) * (1 - eta)),   
            -eta * (1 + xi),                                              
            0.25 * ((1 + xi) * (xi + eta - 1) + (1 + xi) * (1 + eta)),    
            0.5 * (1 - xi**2),                                            
            0.25 * ((1 - xi) * (-xi + eta - 1) + (1 - xi) * (1 + eta)),   
            -eta * (1 - xi)                                               
        ])

        return (N, dN_dxi, dN_deta)