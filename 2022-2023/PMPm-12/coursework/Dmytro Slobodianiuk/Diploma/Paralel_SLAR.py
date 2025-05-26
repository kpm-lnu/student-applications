import numpy as np
import sympy as sm
import math
from func import *
import threading
import time
from SLAR import *



 #def Paralel_solve_bad(self):
        
 #       self.max_it = 100
 #       self.F = sm.Matrix(self.F)
 #       t_arr = self.x_0_arr
 #       x_arr = self.x_0_arr
 #       delta = self.x_0_arr
 #       self.k = 0
 #       self.p = 5

 #       Par_mtrx = []
 #       t = 0
 #       res = []

 #       for i in range(self.p):
 #           #print(i)
 #           thread = threading.Thread(target = self.Par_solver_bad,args=(i,x_arr,res))
 #           thread.start()

 #       for thread in threading.enumerate():
 #           if thread != threading.current_thread():
 #               thread.join()

 #       x_1 = self.x_0_arr
 #       x_2 = sm.Matrix(res)
        
 #       while(self.Norma_vect(x_1,x_2)):

 #           if(self.k > self.max_it):
 #               print("\nЗабагато ітерацій!!!")
 #               break
 #           res = []
 #           for i in range(self.p):
 #               #print(i)
 #               thread = threading.Thread(target = self.Par_solver,args=(i,x_2,res))
 #               thread.start()

 #           for thread in threading.enumerate():
 #               if thread != threading.current_thread():
 #                   thread.join()            
 #           x_1 = x_2
 #           x_2 = sm.Matrix(res)
 #           #print(self.k,"- ітераці:",x_2)
 #           self.x_arr = x_2

 #           self.k += 1
 #           self.Results()

 #       return self.Results()
    
 #   def Norma_vect(self,arr_1,arr_2):
 #       t = 0
 #       for i in range(self.n):
 #           t += (arr_1[i] + arr_2[i])**2
 #       return math.sqrt(t)

    #def Par_solver_bad(self,j,t_arr,res):
    #    t_mtrx = []
    #    for i in range(self.p):
    #        t_mtrx.append(self.F[j*self.p+i])
    #    t_mtrx = sm.Matrix(t_mtrx)

    #    tt = []
    #    if(j == 0):
    #        k_current = sm.sympify("x" + str(self.p+1))
    #        t_mtrx = t_mtrx.subs(k_current,t_arr[self.p]).evalf(5)
    #    elif(j > 0 and j < self.p):
    #        k_current_1 = sm.sympify("x" + str(self.p*j))
    #        t_mtrx = t_mtrx.subs(k_current_1,t_arr[self.p*j-1]).evalf(5)
    #        k_current_2 = sm.sympify("x" + str(self.p*(j + 1)+1))
    #        t_mtrx = t_mtrx.subs(k_current_2,t_arr[self.p*(j + 1)-1]).evalf(5)
    #    elif(j == self.p - 1):
    #        k_current = sm.sympify("x" + str(self.p*j))
    #        t_mtrx = t_mtrx.subs(k_current,t_arr[selp.p*j-1]).evalf(5)

    #    Y = []
    #    for i in range(self.p):
    #        k_current = sm.sympify("x" + str(j*self.p + i+1))
    #        Y.append(k_current)
    #    self.Y = sm.Matrix(Y)

    #    t_jac = t_mtrx.jacobian(self.Y)

        
    #    for i in range(self.p):
    #        k_current = sm.sympify("x" + str(self.p*j + i + 1))
    #        t_jac = t_jac.subs(k_current,t_arr[self.p*j + i])
    #    t_jac = sm.Matrix(t_jac)

    #    t_inv = t_jac.inv()
        
    #    for i in range(self.p):
    #        k_current = sm.sympify("x" + str(self.p*j + i + 1))
    #        t_mtrx = t_mtrx.subs(k_current,t_arr[self.p*j+1])
        
    #    alpha_k = self.Alpha_k(self.k)
    #    alpha_k = sm.Matrix(alpha_k).evalf(5)


    #    #print("t_inv",t_inv)
    #    #print("t_mtrx",t_mtrx)
    #    #print("alpha_k",alpha_k)
    #    for i in range(self.p):
    #        t_mtrx[i] = alpha_k[i]*t_mtrx[i]


    #    delta = -t_inv*t_mtrx
    #    for i in range(self.p):
    #        t = t_arr[self.p*j+i] + delta[i]
    #        res.append(t)

  
    #    return res