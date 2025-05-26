import numpy as np
import sympy as sm
import math

x1,x2,x3,x4,x5,x6,x7,x8,x9,x10,x11,x12,x13,x14,x15,x16,x17,x18,x19,x20 = sm.symbols('x1:21')
y1,y2,y3,y4,y5,y6,y7,y8,y9,y10,y11,y12,y13,y14,y15,y16,y17,y18,y19,y20 = sm.symbols('y1:21')


def Exmpl_1(n):
    F = []
    for i in range(n+1):
        k_plus = sm.sympify("x" + str(i+1))
        k_minus = sm.sympify("x" + str(i-1))
        k_current = sm.sympify("x" + str(i)) 
        if(i==1):
            t = k_current*(0.5*k_current - 3) + 2*k_plus - 1
            F.append(t)
        elif(1<i<n):
            t = k_current*(0.5*k_current - 3) + k_minus + 2*k_plus - 1
            F.append(t)
        elif(i==n):
            t = k_current*(0.5*k_current - 3) - 1 + k_minus
            F.append(t)

    return F


def Exmpl_2(n):
    eqns = [(3 - 2*x1)*x1 - 2*x2 + 1]
    for i in range(2,n+1):
        k_plus = sm.sympify("x" + str(i+1))
        k_minus = sm.sympify("x" + str(i-1))
        k_current = sm.sympify("x" + str(i))  

        t = (3 - 2*k_current)*k_current -k_minus - 2*k_plus + 1

        if(i == n):
            t = (3 - 2*k_current)*k_current - k_minus + 1
        eqns.append(t)
    return eqns


def Print_eq(n,F):
    for i in range(n):
        print(F[i])



def X_0(n):
    x_0_arr = []
    for i in range(1,n+1):
        x_0_arr.append(-1)
    return x_0_arr



def Jacobian(n,F):
    Y = []
    for i in range(1,n+1):
        k_current = sm.sympify("x" + str(i))
        Y.append(k_current)
    Y = sm.Matrix(Y)
    #print(Y)
    #print("Jacobian:\n",F.jacobian(Y))
    return F.jacobian(Y)


def norma(arr):
    t = []
    for i in range(len(arr)):
        t.append(abs(arr[i]))
    return max(t)


def norma_sich(arr):
    t = []
    for i in range(len(arr)):
        t.append(arr[i]**2)
    return(math.sqrt(sum(t)))