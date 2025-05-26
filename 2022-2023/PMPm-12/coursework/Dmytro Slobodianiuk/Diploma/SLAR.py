import numpy as np
import sympy as sm
import math
from func import *
import threading
import time
import random



x1,x2,x3,x4,x5,x6,x7,x8,x9,x10,x11,x12,x13,x14,x15,x16,x17,x18,x19,x20 = sm.symbols('x1:21')
y1,y2,y3,y4,y5,y6,y7,y8,y9,y10,y11,y12,y13,y14,y15,y16,y17,y18,y19,y20 = sm.symbols('y1:21')



class SLAR:
    def __init__(self,n):
        F = []
        x_0_arr = []

        self.n = n
        self.p = 5
        self.eps = 10**(-3)
        self.F = F
        self.x_0_arr = x_0_arr
        self.x_arr = self.x_0_arr
        print("SLAR is created")

    def Example(self,j):
        #F = []
        self.j = j
        if(self.j == 1):
            for i in range(self.n+1):
                k_plus = sm.sympify("x" + str(i+1))
                k_minus = sm.sympify("x" + str(i-1))
                k_current = sm.sympify("x" + str(i)) 
                if(i==1):
                    t = k_current*(0.5*k_current - 3) + 2*k_plus - 1
                    self.F.append(t)
                elif(1<i<self.n):
                    t = k_current*(0.5*k_current - 3) + k_minus + 2*k_plus - 1
                    self.F.append(t)
                elif(i==self.n):
                        t = k_current*(0.5*k_current - 3) - 1 + k_minus
                        self.F.append(t)
        if(self.j == 2):
              self.F = [(3 - 2*x1)*x1 - 2*x2 + 1]
              for i in range(2,self.n+1):
                  k_plus = sm.sympify("x" + str(i+1))
                  k_minus = sm.sympify("x" + str(i-1))
                  k_current = sm.sympify("x" + str(i))  

                  t = (3 - 2*k_current)*k_current -k_minus - 2*k_plus + 1
                  if(i == self.n):
                      t = (3 - 2*k_current)*k_current - k_minus + 1
                  self.F.append(t)
        if(self.j == 3):
            self.F = [(3 - 2*x1)*x1 - 2*x2 + 1]
            for i in range(2,self.n+1):
                  k_plus = sm.sympify("x" + str(i+1))
                  k_minus = sm.sympify("x" + str(i-1))
                  k_current = sm.sympify("x" + str(i))  
                  t = (3 - 2*k_current)*k_current - k_minus -2*k_plus + 1
                  if(i == self.n):
                    t = (3 - 2*k_current)*k_current - k_minus + 1
                  self.F.append(t)
        if(self.j == 4):
            self.F = [3*x1**3 + 2*x2 - 5 + sm.sin(x1 - x2)*sm.sin(x1 + x2)]
            for i in range(2,self.n+1):
                k_plus = sm.sympify("x" + str(i+1))
                k_minus = sm.sympify("x" + str(i-1))
                k_current = sm.sympify("x" + str(i)) 
                t = 3*k_current**3 + 2*k_plus - 5 + sm.sin(k_current - k_plus)*sm.sin(k_current + k_plus) + 4*k_current - k_minus*sm.exp(k_minus - k_current)-3
                if(i == self.n):
                    t = 4*k_current - k_minus*sm.exp(k_minus - k_current)-3
                self.F.append(t)
                     
        #print(self.F)
        return self.F

    def Print_eq(self):
        print("\n")
        for i in range(self.n):
            if(i%self.p == 0):
                print("\n")
            print(self.F[i])

    def Print_eq_par(self,A,m):
        for i in range(m):
            print(A[i])
    #def Print_x(self):
    #    print("\n")
    #    for i in range(self.n):
    #        print(self.x_0_arr[i])

    def X_0(self):
        if(self.j == 1):
            for i in range(1,self.n+1):
                self.x_0_arr.append(-1)
        elif(self.j == 2):
            for i in range(1,self.n+1):
                self.x_0_arr.append(-1)
        elif(self.j == 3):
            for i in range(1,self.n+1):
                self.x_0_arr.append(-1)
        elif(self.j == 4):
            for i in range(1,self.n+1):
                self.x_0_arr.append(-1)
        else:
            for i in range(1,self.n+1):
                self.x_0_arr.append(-1)
        #print(self.x_0_arr)
        self.x_0_arr = sm.Matrix(self.x_0_arr)
        return self.x_0_arr

    def norma(self,arr):
        t = []
        for i in range(len(arr)):
            t.append(abs(arr[i]))
        return max(t)

    def Jacobian(self):
        Y = []
        self.Y = Y
        F_t = self.F
        for i in range(1,self.n+1):
            k_current = sm.sympify("x" + str(i))
            self.Y.append(k_current)
        self.Y = sm.Matrix(Y)
        return F_t.jacobian(self.Y)

    def Results(self):
        print("кількість ітерацій:",self.k)
        #print("значення x:",self.x_arr.evalf(5))
        F_t = self.F
        for i in range(self.n):
            k_current = sm.sympify("x" + str(i+1))
            F_t = F_t.subs(k_current,self.x_arr[i]).evalf(5)
        #print("значення F(x):",F_t)

    def Alpha_k(self,k):
        alpha_k = []
        for i in range(self.p):
            a_t = []
            t = 1/(2**(k))
            a_t.append(t)
            alpha_k.append(a_t)
        return alpha_k
    #послідовний алгоритм
    def Solve(self):
        start_time = time.time()
        self.F = sm.Matrix(self.F)
        t_arr = self.x_0_arr
        x_arr = self.x_0_arr
        delta = self.x_0_arr
        self.k = 0
        F_t = self.F

        while(norma(delta)>self.eps):

            for i in range(self.n):
                #t_arr[i] = t_arr[i]
                x_arr[i] = x_arr[i].evalf(5)
                t_arr[i] = t_arr[i].evalf(5)
            t_arr = x_arr

            #print("\n",x_arr)
            #print(t_arr)

            #print("F_t",F_t)
            #F_jac = Jacobian(self.n,F_t)
            #F_inv = F_jac.inv()

            #for i in range(self.n):
            #    k_current = sm.sympify("x" + str(i+1))
            #    F_inv = F_inv.subs(k_current,t_arr[i])
            #    F_t = F_t.subs(k_current,t_arr[i])

            F_t = self.F
            F_jac = self.Jacobian()
            for i in range(self.n):
                k_current = sm.sympify("x"+str(i+1))
                F_jac = F_jac.subs(k_current,t_arr[i])
                F_t = F_t.subs(k_current,t_arr[i])
            F_inv = F_jac.inv()


            delta = -F_inv*F_t
            x_arr = x_arr+delta
            self.k+=1
            if(norma(t_arr)<self.eps):
                break
        self.x_arr = x_arr
        print("\nПрямий метод:")
        print("Норма:",norma(delta))
        print("--- %s seconds ---"%(time.time() - start_time))
        return self.Results()
    

    #паралельний алгоритм
    def Paralel_solve(self):
        start_time = time.time()
        self.F = sm.Matrix(self.F)
        t_arr = self.x_0_arr
        x_arr = self.x_0_arr
        delta = self.x_0_arr
        self.k = 0
        F_t = self.F

        while(norma(delta)>self.eps):
            for i in range(self.n):
                x_arr[i] = x_arr[i].evalf(5)
                t_arr[i] = t_arr[i].evalf(5)
            t_arr = x_arr

            F_t = self.F
            F_jac = self.Jacobian()

            for i in range(self.n):
                k_current = sm.sympify("x"+str(i+1))
                F_jac = F_jac.subs(k_current,t_arr[i])
                F_t = F_t.subs(k_current,t_arr[i])


            #res = []
            #F_jac = self.Insert_mtrx(F_jac,t_arr)
            #F_jac = sm.Matrix(F_jac)
            #F_t = self.Insert_vctr(F_t,t_arr,res)
            #F_t = sm.Matrix(F_t)




            F_inv = F_jac.inv()


            delta = -F_inv*F_t


            x_arr = x_arr+delta
            self.k+=1
            if(norma(t_arr)<self.eps):
                break
        self.x_arr = x_arr
        print("Норма:",norma(delta))
        print("\nПаралельний метод:")
        print("--- %s seconds ---"%(time.time() - start_time))
        return self.Results()
    
    def Arr_plus(self,arr1,arr2,res):
        for i in range(self.n):
            res.append(arr1[i] + arr2[i])



    #невдалий паралельний алгоритм
    def Paralel_solve_bad(self):
        
        self.max_it = 100
        self.F = sm.Matrix(self.F)
        t_arr = self.x_0_arr
        x_arr = self.x_0_arr
        delta = self.x_0_arr
        self.k = 0
        self.p = 5

        Par_mtrx = []
        t = 0
        res = []

        for i in range(self.p):
            #print(i)
            thread = threading.Thread(target = self.Par_solver_bad,args=(i,x_arr,res))
            thread.start()

        for thread in threading.enumerate():
            if thread != threading.current_thread():
                thread.join()

        x_1 = self.x_0_arr
        x_2 = sm.Matrix(res)
        
        while(self.Norma_vect(x_1,x_2)):

            if(self.k > self.max_it):
                print("\nЗабагато ітерацій!!!")
                break
            res = []
            for i in range(self.p):
                #print(i)
                thread = threading.Thread(target = self.Par_solver,args=(i,x_2,res))
                thread.start()

            for thread in threading.enumerate():
                if thread != threading.current_thread():
                    thread.join()            
            x_1 = x_2
            x_2 = sm.Matrix(res)
            #print(self.k,"- ітераці:",x_2)
            self.x_arr = x_2

            

            self.k += 1
            self.Results()


        return self.Results()
    
    def Norma_vect(self,arr_1,arr_2):
        t = 0
        for i in range(self.n):
            t += (arr_1[i] + arr_2[i])**2
        return math.sqrt(t)

    def Par_solver_bad(self,j,t_arr,res):
        t_mtrx = []
        for i in range(self.p):
            t_mtrx.append(self.F[j*self.p+i])
        t_mtrx = sm.Matrix(t_mtrx)

        tt = []
        if(j == 0):
            k_current = sm.sympify("x" + str(self.p+1))
            t_mtrx = t_mtrx.subs(k_current,t_arr[self.p]).evalf(5)
        elif(j > 0 and j < self.p):
            k_current_1 = sm.sympify("x" + str(self.p*j))
            t_mtrx = t_mtrx.subs(k_current_1,t_arr[self.p*j-1]).evalf(5)
            k_current_2 = sm.sympify("x" + str(self.p*(j + 1)+1))
            t_mtrx = t_mtrx.subs(k_current_2,t_arr[self.p*(j + 1)-1]).evalf(5)
        elif(j == self.p - 1):
            k_current = sm.sympify("x" + str(self.p*j))
            t_mtrx = t_mtrx.subs(k_current,t_arr[selp.p*j-1]).evalf(5)

        Y = []
        for i in range(self.p):
            k_current = sm.sympify("x" + str(j*self.p + i+1))
            Y.append(k_current)
        self.Y = sm.Matrix(Y)

        t_jac = t_mtrx.jacobian(self.Y)

        
        for i in range(self.p):
            k_current = sm.sympify("x" + str(self.p*j + i + 1))
            t_jac = t_jac.subs(k_current,t_arr[self.p*j + i])
        t_jac = sm.Matrix(t_jac)

        t_inv = t_jac.inv()
        
        for i in range(self.p):
            k_current = sm.sympify("x" + str(self.p*j + i + 1))
            t_mtrx = t_mtrx.subs(k_current,t_arr[self.p*j+1])
        
        alpha_k = self.Alpha_k(self.k)
        alpha_k = sm.Matrix(alpha_k).evalf(5)


        #print("t_inv",t_inv)
        #print("t_mtrx",t_mtrx)
        #print("alpha_k",alpha_k)
        for i in range(self.p):
            t_mtrx[i] = alpha_k[i]*t_mtrx[i]


        delta = -t_inv*t_mtrx
        for i in range(self.p):
            t = t_arr[self.p*j+i] + delta[i]
            res.append(t)

  
        return res



        #if(self.j == 3):
        #    for i in range(self.n+1):
        #        k_i = sm.sympify("x" + str(i+1))
        #        t = self.n
        #        q = 0
        #        for o in range(1,self.n):
        #            #k_j = sm.sympify("x" + str(o+1))
        #            #q += sm.cos(k_j) + i*(1-sm.cos(k_i)) - sm.sin(k_i)
        #            k_j = sm.sympify("x" + str(o+1))
        #            q += sm.cos(k_j)
        #        t = t - q + i*(1 - sm.cos(k_i)) - sm.sin(k_i)
        #        self.F.append(t)

        #def Insert_mtrx(self,F_t,x_arr):
    #    res = []
    #    #print("Insert_mtrx wordked")
    #    for i in range(self.p):
    #        thread = threading.Thread(target = self.Insert_vctr,args=(F_t[i],x_arr,res))
    #        thread.start()
    #    for thread in threading.enumerate():
    #        if thread != threading.current_thread():
    #            thread.join()

    #    res = sm.Matrix(res)
    #    return res

    #def Insert_vctr(self,vctr,arr,res):

    #    for i in range(self.n):
    #        k_current = sm.sympify("x" + str(i+1))
    #        vctr = vctr.subs(k_current,arr[i]).evalf(5)
    #    res.append(vctr)
    #    return res





    #def Insert_mtrx(self,F_t,x_arr):
    #    res = []
    #    #print("Insert_mtrx wordked")
    #    for i in range(self.p):
    #        thread = threading.Thread(target = self.Insert_vctr,args=(F_t[i],x_arr,res))
    #        thread.start()
    #    for thread in threading.enumerate():
    #        if thread != threading.current_thread():
    #            thread.join()

    #    res = sm.Matrix(res)
    #    return res

    #def Insert_vctr(self,vctr,arr,res):

    #    for i in range(self.n):
    #        k_current = sm.sympify("x" + str(i+1))
    #        vctr = vctr.subs(k_current,arr[i]).evalf(5)
    #    res.append(vctr)
    #    return res




            #for i in range(self.p):
        #    t_res = []
        #    for j in range(self.p):
        #        t_res.append(self.F[t])
        #        t += 1
        #    Par_mtrx.append(t_res)
        #    t_res = sm.Matrix(t_res)

        #    thread = threading.Thread(target=self.Par_solver,args=(t_res,x_arr,i,res))
        #    thread.start()

        #    print("thread(",i,"):",thread)
    
    #def Insert_stream(self,F_t,x_arr):
    #    res = []
    #    for i in range(self.p):
    #        thread = threading.Thread(target=self.Insert,args=(F_t[i],i,x_arr,res))
    #        thread.start()
    #    for thread in threading.enumerate():
    #        if thread != threading.current_thread():
    #            thread.join()
    #    return res

    #def Insert(self,arr,i,x_arr,res):
    #    for i in range(self.n):
    #        k_current = sm.sympify("x" + str(i+1))
    #        arr = arr.subs(k_current,x_arr[i]).evalf(5)
    #    res.append(arr)

    #def Insert_jac(self,F_t,x_arr):
    #    res = []
    #    F_t = self.F
    #    F_jac = Jacobian(self.n,F_t)

    #    print(F_jac)














    #def Multi_subs(self,arr,i,res):
    #    for i in range(self.n):
    #        k_current = sm.sympify("x" + str(i+1))
    #        arr = arr.subs(k_current,self.x_arr[i]).evalf(5)
    #    res.append(arr)

    #def Thread_test(self):
    #    res = []
    #    F_t = sm.sympify(self.F)
    #    for i in range(self.n):
    #        thread = threading.Thread(target = self.Multi_subs,args=(F_t[i],i,res))
    #        thread.start()

    #    for thread in threading.enumerate():
    #        if thread != threading.current_thread():
    #            thread.join()
    #    print(self.x_arr)
    #    print(res)














    #def Test_par(self,F_t,i,res):
    #    #F_t = sm.Matrix(self.F)
    #    k_current = sm.sympify("x" + str(i))
    #    res.append(F_t.subs(k_current,self.x_arr[i]))
    #    return res
    #def Input_x(self,F_t):
    #    print("Input_x worked")
    #    F_t = sm.Matrix(self.F)
    #    for i in range(self.n):
    #        F_t = self.Test_par(F_t,i)
    #    #self.Print_eq_par(F_t,self.n)
    #def Thread_test(self):
    #    q = 6
    #    F_t = sm.Matrix(self.F)
    #    res = []
    #    for i in range(self.n):
    #        thread = threading.Thread(target=self.Test_par,args=(F_t,i,res))
    #        thread.start()
    #    for thread in threading.enumerate():
    #        if thread != threading.current_thread():
    #            thread.join()
    #    F_t = res
    #    print(res)
    #    self.Print_eq_par(F_t,self.n)