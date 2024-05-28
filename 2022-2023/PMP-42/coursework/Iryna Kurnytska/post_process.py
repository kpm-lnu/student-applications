import numpy as np
import math

def post_process(NL,EL,ENL):
    NoE= np.size(EL,0)
    NPE= np.size(EL,1)
    PD=np.size(NL,1)

    scale=1 # Magnify the deflection

    disp, stress, strain = element_post_process(NL,EL,ENL)


    stress_xx=np.zeros([NPE,NoE])
    stress_xy=np.zeros([NPE,NoE])
    stress_yx=np.zeros([NPE,NoE])
    stress_yy=np.zeros([NPE,NoE])

    strain_xx=np.zeros([NPE,NoE])
    strain_xy=np.zeros([NPE,NoE])
    strain_yx=np.zeros([NPE,NoE])
    strain_yy=np.zeros([NPE,NoE])

    disp_x=np.zeros([NPE,NoE])
    disp_y=np.zeros([NPE,NoE])

    X=np.zeros([NPE,NoE])
    Y=np.zeros([NPE,NoE])

    if NPE in [3,4]:

        X=ENL[EL-1,0] + scale* ENL[EL-1, 4*PD]
        Y=ENL[EL-1,1] + scale* ENL[EL-1, 4*PD+1]

        X=X.T
        Y=Y.T
        stress_xx[:,:]= stress[:,:,0,0].T
        stress_xy[:,:]= stress[:,:,0,1].T
        stress_yx[:,:]= stress[:,:,1,0].T
        stress_yy[:,:]= stress[:,:,1,1].T

        strain_xx[:,:]= strain[:,:,0,0].T
        strain_xy[:,:]= strain[:,:,0,1].T
        strain_yx[:,:]= strain[:,:,1,0].T
        strain_yy[:,:]= strain[:,:,1,1].T     

        disp_x= disp[:,:,0,0].T
        disp_y=disp[:,:,1,0].T   


    return (stress_xx, stress_xy, stress_yx, stress_yy, strain_xx,strain_xy,
 strain_yx,strain_yy,disp_x,disp_y,X,Y)

def element_post_process(NL,EL,ENL):

    PD=np.size(NL,1)
    NoE=np.size(EL,0) # specifies the element I'm looking at
    NPE=np.size(EL,1)  # specifies the node of  the element I'm looking at

    if NPE==3:
        GPE=1
    if NPE==4:
        GPE=4
    disp=np.zeros([NoE,NPE,PD,1])


    stress=np.zeros([NoE,GPE,PD,PD])
    strain=np.zeros([NoE,GPE,PD,PD])

    for e in  range(1,NoE+1):
        nl =EL [e-1,0:NPE]
        for i in range(1,NPE+1):
            for j in range(1,PD+1):
                disp[e-1,i-1,j-1,0]=ENL[nl[i-1]-1, 4*PD+j-1]
        
        x=np.zeros([NPE,PD])
        x[0:NPE, 0:PD]= NL[nl[0:NPE]-1, 0:PD]
#specify the displacements for these corners 
        u=np.zeros([PD,NPE])
        for i in range(1,NPE+1):
            for j in range(1,PD+1):
                u[j-1,i-1] = ENL[nl[i-1]-1,4*PD+j-1]

        coor=x.T

        for gp in range(1, GPE+1):
            #strain for each Gauss point (2x2 matrix)
            epsilon=np.zeros([PD,PD])

            for i in range(1,NPE+1):    
                J=np.zeros([PD,PD])

                grad=np.zeros([PD,NPE])   
                (xi, eta,alpha) = GaussPoint(NPE,GPE,gp)   

                grad_nat=grad_N_nat(NPE,xi, eta)   
                J=coor @ grad_nat.T

                grad= np.linalg.inv(J).T @ grad_nat

                #calculate strain 
                # define dyadic in another function
                epsilon= epsilon+1/2 *(dyad(grad[:,i-1], u[:,i-1])+ dyad(u[:,i-1], grad [:,i-1]))

                #initiliaze stress as 2x2 matrix

            sigma=np.zeros([PD,PD]) 
            #sigma= E * epsilon
            for a in range (1,PD+1):
                for b in range (1,PD+1):
                    for c in range (1,PD+1):
                        for d in range (1,PD+1):
                            sigma[a-1,b-1] = sigma[a-1,b-1]+ constitutive(a,b,c,d) * epsilon[c-1,d-1]

            for a in range (1,PD+1):
                for b in range (1,PD+1):
                    strain[e-1, gp-1, a-1,b-1] = epsilon[a-1,b-1]
                    stress[e-1,gp-1,a-1,b-1]  = sigma [a-1,b-1]            


    return disp, stress, strain

def dyad(u,v):
    u=u.reshape(len(v),1)
    v=v.reshape(len(v),1)
    PD=2
    A = u @ v.T
    return A



def GaussPoint(NPE,GPE,gp):
    if NPE==3:
        if GPE==1:
            if gp==1:
                xi=1/3
                eta=1/3
                alpha=1
    elif NPE==4:
        if GPE==1:
            if gp==1:
                xi= 0
                eta= 0
                alpha= 4

        elif GPE==4:
            if gp==1:

                xi= -1/math.sqrt(3)
                eta= -1/math.sqrt(3)
                alpha= 1

            elif gp==2:

                xi= 1/math.sqrt(3)
                eta= -1/math.sqrt(3)
                alpha= 1


            elif gp==3:
                xi= 1/math.sqrt(3)
                eta= 1/math.sqrt(3)
                alpha= 1
            
            elif gp==4:
                xi= -1/math.sqrt(3)
                eta= 1/math.sqrt(3)
                alpha= 1

    return xi, eta, alpha


def grad_N_nat(NPE,xi,eta): #похідні формувальних функцій у вузлах скінченного елементу. 
    
    PD=2
    result=np.zeros([PD,NPE])
    if NPE==3:
        result[0,0]=1
        result[0,1]=0
        result[0,2]=-1
       

        result[1,0]=0
        result[1,1]=1
        result[1,2]=-1
   


    if NPE==4:
        result[0,0]=-1/4*(1-eta)
        result[0,1]=1/4*(1-eta)
        result[0,2]=1/4*(1+eta)
        result[0,3]=-1/4*(1+eta)

        result[1,0]=-1/4*(1-xi)
        result[1,1]=-1/4*(1+xi)
        result[1,2]=1/4*(1+xi)
        result[1,3]=1/4*(1-xi)

    return result

def constitutive(i,j,k,l):

    E= 8/3
    nu=1/3

    C=(E/(2*(1+nu))) * (delta(i,l)*delta(j,k) + delta(i,k) * delta(j,l)) + (E*nu)/(1-nu**2)* delta(i,j)*delta(k,l)

    return C

def delta(i,j):

    if i==j:
        delta=1
    else: 
        delta=0

    return delta