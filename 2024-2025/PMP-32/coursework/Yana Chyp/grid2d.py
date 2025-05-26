#
# import numpy as np
# import matplotlib.pyplot as plt
#
# def uniform_mesh(d1, d2, p, m, element_type):
#     PD=2 #простір
#
#     q=np.array([[0,0],[d1,0],[0,d2],[d1,d2]]) #4 corners
#
#     NoN= (p+1)*(m+1)  #number of nodes
#     NoE=p*m  #number of elements
#     NPE = 4 #бо в нас прямокутник, якщо хочемо трикутник міняємо на 3 nodes in element
#
#     NL = np.zeros([NoN, PD])
#     a = (q[1,0]-q[0,0])/p  #increments in the horizontal direction
#     b = (q[2,1]-q[0,1])/m #increments in the vertical direction
#
#     n = 0 #go throught the row in the note NL
#
#     for i in range(1, m+2):
#         for j in range(1, p+2):
#             NL[n,0] = q[0,0]+(j-1)*a  #for x values
#             NL[n,1] = q[0,1]+(i-1)*b  #for y values
#
#             n+=1
#
#     #elements
#     EL = np.zeros([NoE, NPE], dtype=int)
#
#     for i in range(1, m+1):
#         for j in range(1, p+1):
#             if j==1:
#                 EL[(i-1)*p+j-1, 0] = (i-1)*(p+1)+j
#                 EL[(i-1)*p+j-1, 1] = EL[(i-1)*p+j-1, 0] + 1
#                 EL[(i-1)*p+j-1, 3] = EL[(i-1)*p+j-1, 0] + (p+1)
#                 EL[(i-1)*p+j-1, 2] = EL[(i-1)*p+j-1, 3] + 1
#             else:
#                 EL[(i-1)*p+j-1, 0] = EL[(i-1)*p+j-2, 1]
#                 EL[(i-1)*p+j-1, 3] = EL[(i-1)*p+j-2, 2]
#                 EL[(i-1)*p+j-1, 1] = EL[(i-1)*p+j-1, 0] + 1
#                 EL[(i-1)*p+j-1, 2] = EL[(i-1)*p+j-1, 3] + 1
#     if element_type == 'D2TR3N':
#         NPE_new=3 #Triangular Element
#         NoE_new = 2*NoE #Every element will be divided into two triangles
#         EL_new = np.zeros([NoE_new, NPE_new]) #New size for El
#
#         for i in range(1, NoE+1):
#             #for the first triangular element
#             EL_new[2*(i-1), 0] = EL[i-1, 0]
#             EL_new[2*(i-1), 1] = EL[i-1, 1]
#             EL_new[2*(i-1), 2] = EL[i-1, 2]
#             # for the second triangular element
#             EL_new[2*(i-1), 0] = EL[i-1, 0]
#             EL_new[2*(i-1), 1] = EL[i-1, 1]
#             EL_new[2*(i-1), 2] = EL[i-1, 2]
#         EL=EL_new
#
#     EL = EL.astype(int)
#
#     return NL, EL
# def main():
#     d1=1
#     d2=1
#     p=4
#     m=3
#     element_type = 'D2QU4N'
#
#     NL, EL=uniform_mesh(d1, d2, p, m, element_type)
#
#     NoN = np.size(NL, 0)
#     NoE = np.size(EL, 0)
#
#     plt.figure(1)
#
#     count = 1 #Annotate node numbers
#     for i in range(0, NoN):
#         plt.annotate(count, xy=(NL[i, 0], NL[i, 1]))
#         count +=1
#     if element_type == 'D2QU4N':
#         count2=1 #Annotate element numbers
#         for j in range(0, NoE):
#             plt.annotate(count2, xy=((NL[EL[j,0]-1,0]+NL[EL[j,1]-1,0]+NL[EL[j,2]-1,0]+NL[EL[j,3]-1,0])/4,
#                                      (NL[EL[j,0]-1,1]+NL[EL[j,1]-1,1]+NL[EL[j,2]-1,1]+NL[EL[j,3]-1,1])/4),
#                                     c = 'blue')
#             count2+=1
#
#         #Plot lines
#         x0,y0=NL[EL[:,0]-1,0], NL[EL[:,0]-1,1]
#         x1,y1=NL[EL[:,1]-1,0], NL[EL[:,1]-1,1]
#         x2,y2=NL[EL[:,2]-1,0], NL[EL[:,2]-1,1]
#         x3,y3=NL[EL[:,3]-1,0], NL[EL[:,3]-1,1]
#         plt.plot(np.array([x0,x1]), np.array([y0, y1]), 'red', linewidth=3)
#         plt.plot(np.array([x1,x2]), np.array([y1, y2]), 'red', linewidth=3)
#         plt.plot(np.array([x2,x3]), np.array([y2, y3]), 'red', linewidth=3)
#         plt.plot(np.array([x3,x0]), np.array([y3, y0]), 'red', linewidth=3)
#
#     if element_type == 'D2TR3N':
#         count2=1 #Annotate element numbers
#         for j in range(0, NoE):
#             plt.annotate(count2, xy=((NL[EL[j,0]-1,0]+NL[EL[j,1]-1,0]+NL[EL[j,2]-1,0])/3,
#                                      (NL[EL[j,0]-1,1]+NL[EL[j,1]-1,1]+NL[EL[j,2]-1,1])/3),
#                                     c = 'blue')
#             count2+=1
#
#         #Plot lines
#         x0,y0=NL[EL[:,0]-1,0], NL[EL[:,0]-1,1]
#         x1,y1=NL[EL[:,1]-1,0], NL[EL[:,1]-1,1]
#         x2,y2=NL[EL[:,2]-1,0], NL[EL[:,2]-1,1]
#         plt.plot(np.array([x0,x1]), np.array([y0, y1]), 'red', linewith=3)
#         plt.plot(np.array([x1,x2]), np.array([y1, y2]), 'red', linewith=3)
#         plt.plot(np.array([x2,x0]), np.array([y2, y0]), 'red', linewith=3)
#
import numpy as np
import matplotlib.pyplot as plt

def uniform_mesh(d1, d2, p, m, element_type):
    PD = 2  # Простір (x, y)
    q = np.array([[0, 0], [d1, 0], [0, d2], [d1, d2]])  # 4 кути прямокутника

    NoN = (p + 1) * (m + 1)  # Кількість вузлів
    NoE = p * m  # Кількість елементів (для прямокутників)
    NPE = 4 if element_type == 'D2QU4N' else 3  # Кількість вузлів на елемент

    NL = np.zeros([NoN, PD])
    a = (q[1, 0] - q[0, 0]) / p  # Інкременти по x
    b = (q[2, 1] - q[0, 1]) / m  # Інкременти по y

    n = 0
    for i in range(m + 1):
        for j in range(p + 1):
            NL[n, 0] = q[0, 0] + j * a  # Значення x
            NL[n, 1] = q[0, 1] + i * b  # Значення y
            n += 1

    if element_type == 'D2QU4N':  # Прямокутники
        EL = np.zeros([NoE, NPE], dtype=int)
        for i in range(m):
            for j in range(p):
                n1 = i * (p + 1) + j
                n2 = n1 + 1
                n3 = n1 + (p + 1)
                n4 = n3 + 1
                EL[i * p + j] = [n1, n2, n4, n3]
    elif element_type == 'D2TR3N':  # Трикутники
        EL = np.zeros([NoE * 2, NPE], dtype=int)  # Подвійна кількість елементів для трикутників
        e = 0
        for i in range(m):
            for j in range(p):
                n1 = i * (p + 1) + j
                n2 = n1 + 1
                n3 = n1 + (p + 1)
                n4 = n3 + 1
                # Перший трикутник
                EL[e] = [n1, n2, n4]
                e += 1
                # Другий трикутник
                EL[e] = [n1, n4, n3]
                e += 1
    else:
        raise ValueError("Неправильний тип елемента. Використовуйте 'D2QU4N' або 'D2TR3N'.")

    return NL, EL

def main():
    d1 = 1  # Довжина по x
    d2 = 1  # Довжина по y
    p = 4  # Поділ по x
    m = 5  # Поділ по y
    element_type = 'D2QU4N'  # Тип елементу ('D2QU4N' або 'D2TR3N')

    NL, EL = uniform_mesh(d1, d2, p, m, element_type)

    NoN = NL.shape[0]  # Кількість вузлів
    NoE = EL.shape[0]  # Кількість елементів

    plt.figure(figsize=(8, 8))

    # Анотація вузлів
    for i in range(NoN):
        plt.scatter(NL[i, 0], NL[i, 1], color='black')  # Вузли
        plt.text(NL[i, 0], NL[i, 1], str(i + 1), color='red', fontsize=8)

    # Побудова елементів
    for j in range(NoE):
        nodes = EL[j]
        x = NL[nodes, 0]
        y = NL[nodes, 1]
        x = np.append(x, x[0])  # Замкнути контур
        y = np.append(y, y[0])
        plt.plot(x, y, color='blue')
        # Анотація елемента
        cx = np.mean(x[:-1])
        cy = np.mean(y[:-1])
        plt.text(cx, cy, str(j + 1), color='green', fontsize=10)

    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title(f'Сітка: {element_type}')
    plt.grid()
    plt.axis('equal')
    plt.show()

if __name__ == "__main__":
    main()
