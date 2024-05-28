import numpy as np
import math

def void_mesh(d1,d2,p,m,R,element_type):
    PD=2
    q = np.array([[0,0],[d1,0],[0,d2],[d1,d2]]) #координати області (прямокутник\квадрат) 4 corners
    NoN=2*(p+1)*(m+1) + 2*(p-1)*(m+1)
    NoE=4*p*m 
    NPE=4 # for D2QU4N


    #Nodes
    NL=np.zeros([NoN,PD])

    a=(q[1,0]-q[0,0])/p # increment of the horizontal direction
    b=(q[2,1]-q[0,1])/p # increment of the vertical direction

    coor11=np.zeros([(p+1)*(m+1),PD])

    for i in range(1,p+2):
        coor11[i-1,0]=q[0,0]+ (i-1)*a
        coor11[i-1,1]=q[0,1]


    for i in range(1,p+2):
        coor11[m*(p+1)+i-1,0]= R*np.cos((5*math.pi/4)+(i-1)*((math.pi/2)/p))+d1/2

        coor11[m*(p+1)+i-1,1]= R*np.sin((5*math.pi/4)+(i-1)*((math.pi/2)/p))+d2/2

    for i in range(1,m):
        for j in range(1,p+2):
            dx=(coor11[m*(p+1)+j-1,0] -coor11[j-1,0] )/m
            dy=(coor11[m*(p+1)+j-1,1] -coor11[j-1,1] )/m

            coor11[i*(p+1)+j-1,0]= coor11[(i-1)*(p+1)+j-1,0]+dx
            coor11[i*(p+1)+j-1,1]= coor11[(i-1)*(p+1)+j-1,1]+dy


    coor22=np.zeros([(p+1)*(m+1),PD])
    
    for i in range(1,p+2):
        coor22[i-1,0]=q[2,0]+ (i-1)*a

        coor22[i-1,1]=q[2,1]


    for i in range(1,p+2):
        coor22[m*(p+1)+i-1,0]= R*np.cos((3*math.pi/4)-(i-1)*((math.pi/2)/p))+d1/2

        coor22[m*(p+1)+i-1,1]= R*np.sin((3*math.pi/4)-(i-1)*((math.pi/2)/p))+d2/2

    for i in range(1,m):
        for j in range(1,p+2):
            dx=(coor22[m*(p+1)+j-1,0] -coor22[j-1,0] )/m
            dy=(coor22[m*(p+1)+j-1,1] -coor22[j-1,1] )/m

            coor22[i*(p+1)+j-1,0]= coor22[(i-1)*(p+1)+j-1,0]+dx
            coor22[i*(p+1)+j-1,1]= coor22[(i-1)*(p+1)+j-1,1]+dy

    coor33=np.zeros([(p-1)*(m+1),PD])
    
    for i in range(1,p):
        coor33[i-1,0]=q[0,0]
        coor33[i-1,1]=q[0,1]+i*b


    for i in range(1,p):
        coor33[m*(p-1)+i-1,0]= R*np.cos((5*math.pi/4)-(i)*((math.pi/2)/p))+d1/2

        coor33[m*(p-1)+i-1,1]= R*np.sin((5*math.pi/4)-(i)*((math.pi/2)/p))+d2/2

    for i in range(1,m): 
        for j in range(1,p):
            dx=(coor33[m*(p-1)+j-1,0] -coor33[j-1,0] )/m
            dy=(coor33[m*(p-1)+j-1,1] -coor33[j-1,1] )/m

            coor33[i*(p-1)+j-1,0]= coor33[(i-1)*(p-1)+j-1,0]+dx
            coor33[i*(p-1)+j-1,1]= coor33[(i-1)*(p-1)+j-1,1]+dy

    coor44=np.zeros([(p-1)*(m+1),PD])
    
    for i in range(1,p):
        coor44[i-1,0]=q[1,0]
        coor44[i-1,1]=q[1,1]+i*b


    for i in range(1,p):
        coor44[m*(p-1)+i-1,0]= R*np.cos((7*math.pi/4)+(i)*((math.pi/2)/p))+d1/2

        coor44[m*(p-1)+i-1,1]= R*np.sin((7*math.pi/4)+(i)*((math.pi/2)/p))+d2/2

    for i in range(1,m): #!
        for j in range(1,p):
            dx=(coor44[m*(p-1)+j-1,0] -coor44[j-1,0] )/m
            dy=(coor44[m*(p-1)+j-1,1] -coor44[j-1,1] )/m

            coor44[i*(p-1)+j-1,0]= coor44[(i-1)*(p-1)+j-1,0]+dx
            coor44[i*(p-1)+j-1,1]= coor44[(i-1)*(p-1)+j-1,1]+dy

#Reoderinring the nodes
    for i in range(1,m+2):
        NL[(i-1)*4*p:i*4*p,:] =np.vstack([coor11[(i-1)*(p+1):(i)*(p+1),:], 
                                          coor44[(i-1)*(p-1):(i)*(p-1),:],
                                          np.flipud(coor22[(i-1)*(p+1):(i)*(p+1),:]),
                                          np.flipud(coor33[(i-1)*(p-1):(i)*(p-1),:])])

   #Elements
    EL=np.zeros([NoE,NPE]) 

    for i in range(1,m+1):
        for j in range (1,4*p+1):

            if j==1:

                EL[(i-1)*(4*p)+j-1,0]=(i-1)*(4*p)+j
                EL[(i-1)*(4*p)+j-1,1]=EL[(i-1)*(4*p)+j-1,0]+1
                EL[(i-1)*(4*p)+j-1,3]=EL[(i-1)*(4*p)+j-1,0]+4*p
                EL[(i-1)*(4*p)+j-1,2]=EL[(i-1)*(4*p)+j-1,3]+1

            elif j==4*p:
                EL[(i-1)*(4*p)+j-1,0]= i*(4*p)
                EL[(i-1)*(4*p)+j-1,1]=(i-1)*(4*p) +1
                EL[(i-1)*(4*p)+j-1,2]=EL[(i-1)*(4*p)+j-1,0]+1
                EL[(i-1)*(4*p)+j-1,3]=EL[(i-1)*(4*p)+j-1,0]+4*p

            else:
                EL[(i-1)*(4*p)+j-1,0]=EL[(i-1)*(4*p)+j-2,1]
                EL[(i-1)*(4*p)+j-1,3]=EL[(i-1)*(4*p)+j-2,2]
                EL[(i-1)*(4*p)+j-1,2]=EL[(i-1)*(4*p)+j-1,3]+1
                EL[(i-1)*(4*p)+j-1,1]=EL[(i-1)*(4*p)+j-1,0]+1

    if element_type=="D2TR3N":
        NPE_new=3 #triangular element 
        NoE_new=2*NoE # every rectangular will be divided into 2 pieces
        EL_new=np.zeros([NoE_new,NPE_new]) # new size of the elelement list

        for i in range(1,NoE+1):
            # for the first tringular element
            EL_new[2*(i-1),0]=EL[i-1,0]
            EL_new[2*(i-1),1]=EL[i-1,1]
            EL_new[2*(i-1),2]=EL[i-1,2]
            # for the second tringular element
            EL_new[2*(i-1)+1,0]=EL[i-1,0]
            EL_new[2*(i-1)+1,1]=EL[i-1,2]
            EL_new[2*(i-1)+1,2]=EL[i-1,3]
            
        EL=EL_new


    EL=EL.astype(int)   
    return NL, EL
    
   