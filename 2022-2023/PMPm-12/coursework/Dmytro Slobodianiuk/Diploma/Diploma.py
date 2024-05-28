import numpy as np
import sympy as sm
import math
import threading
import time
from func import *
from SLAR import *
from Paralel_SLAR import *

#start_time = time.time()

x1,x2,x3,x4,x5,x6,x7,x8,x9,x10,x11,x12,x13,x14,x15,x16,x17,x18,x19,x20 = sm.symbols('x1:21')
y1,y2,y3,y4,y5,y6,y7,y8,y9,y10,y11,y12,y13,y14,y15,y16,y17,y18,y19,y20 = sm.symbols('y1:21')

j = 4

#n = int(input("n = "))
n = 100
p = 5

a = -1
b = 1
eps = 10**(-3)
it_max = 100



s1 = SLAR(n)
s1.Example(j)
#s1.Print_eq()
s1.X_0()



s1.Solve()
s1.Paralel_solve()
#print("\n\n")

#print("--- %s seconds ---"%(time.time() - start_time))





#s1.Solve()
#print("\n\n")


#s1 = SLAR(n)
#s1.Example(j)
#s1.X_0()

#s1.Thread_test()



#F = Exmpl_1(n)
#F = Exmpl_2(n)
#Print_eq(n,F)

#x_0_arr = X_0(n)
#x_0_arr = sm.Matrix(x_0_arr)
#print("X_0:\n",x_0_arr)

#eps = 10**(-3)
#print(eps)
#F = sm.Matrix(F)

#t_arr = x_0_arr
#x_arr = x_0_arr
#delta = x_0_arr
#k = 0

#while(norma(delta)>eps):
#    print("\n",x_arr)
#    print(t_arr)
#    t_arr = x_arr

#    for i in range(n):
#        x_arr[i] = x_arr[i].evalf(5)
#        t_arr[i] = t_arr[i].evalf(5)
#    F_t = F
#    print("F_t",F_t)
#    F_jac = Jacobian(n,F_t)
#    F_inv = F_jac.inv()

#    for i in range(n):
#        k_current = sm.sympify("x" + str(i+1))
#        F_inv = F_inv.subs(k_current,t_arr[i])
#        F_t = F_t.subs(k_current,t_arr[i])

#    delta = -F_inv*F_t
#    x_arr = x_arr+delta
#    k += 1
#    if(norma(t_arr)<eps):
#        break
   

#print("\nкількість ітерацій:",k)
#print("значення х:",x_arr)
#F_t = F
#for i in range(n):
#    k_current = sm.sympify("x" + str(i+1))
#    F_t = F_t.subs(k_current,x_arr[i])
#print("значення F(x)",F_t)





##F_jac = Jacobian(n,F_t)
##F_inv = F_jac.inv()
##print("Inverse:\n",F_inv)

##for i in range(n):
##    k_current = sm.sympify("x" + str(i+1))
##    F_inv = F_inv.subs(k_current,x_0_arr[i])
##    F_t = F_t.subs(k_current,x_0_arr[i])

##delta = -F_inv*F_t
##print(delta)

##x_1_arr = x_0_arr + delta

##print(x_1_arr)





#F_t = F_t.subs(x0,x_0_arr[0])


#F_jac = Jacobian(n,F_t)
#F_inv = F_jac.inv()

#for i in range(1,n+1):
#    k_current = sm.sympify("x" + str(i))
#    F_inv = F_inv.subs(k_current,x_0_arr[i])
#    F_t = F_t.subs(k_current,x_0_arr[i])

#print(F_inv)
#print(F_t)
#delta  = F_inv*F_t
#x_1_arr = []


#for i in range(n+1):
#    x_1_arr.append(x_0_arr[i]+delta[i])



#print("\n",x_0_arr)
#print("\n",x_1_arr)




#F = sm.Matrix(F)
#F_origin = F
#print(F)
#for i in range(n+1):
#    k_current = sm.sympify("x" + str(i)) 
#    F = F.subs(k_current,x_0_arr[i])

#Print_eq(n,F)


#y_arr = []
#for i in range(n):
#    t = sm.sympify("x" + str(i+1))
#    y_arr.append(t)
#Y = sm.Matrix(y_arr)


#F_jac = Jacobian(n,F_origin)
#print(F_jac)
#F_jac = F_jac.inv()



#while(norma(t)>eps):
