import numpy as np
from numpy.polynomial import Polynomial
from numpy.polynomial.legendre import Legendre
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy import integrate
from tkinter import *

window=Tk()
lbl_eps=Label(window, text="ϵ")
lbl_eps.place(x=130, y=40)
entry_eps=Entry(window, width=50)
entry_eps.place(x=280, y=40)
lbl_b=Label(window, text="b")
lbl_b.place(x=130, y=80)
entry_b=Entry(window, width=50)
entry_b.place(x=280, y=80)
lbl_c=Label(window, text="c")
lbl_c.place(x=130, y=120)
entry_c=Entry(window, width=50)
entry_c.place(x=280, y=120)
lbl_l=Label(window, text="x_max")
lbl_l.place(x=130, y=160)
entry_l=Entry(window, width=50)
entry_l.place(x=280, y=160)
lbl_t=Label(window, text="T")
lbl_t.place(x=130, y=200)
entry_t=Entry(window, width=50)
entry_t.place(x=280, y=200)
lbl_f=Label(window, text="f")
lbl_f.place(x=130, y=240)
entry_f=Entry(window, width=50)
entry_f.place(x=280, y=240)
lbl_boundary_left=Label(window, text="boundary_left")
lbl_boundary_left.place(x=130, y=280)
entry_boundary_left=Entry(window, width=50)
entry_boundary_left.place(x=280, y=280)
lbl_boundary_right=Label(window, text="boundary_right")
lbl_boundary_right.place(x=130, y=320)
entry_boundary_right=Entry(window, width=50)
entry_boundary_right.place(x=280, y=320)
lbl_boundary_left_der=Label(window, text="boundary_left_deriv")
lbl_boundary_left_der.place(x=130, y=360)
entry_boundary_left_der=Entry(window, width=50)
entry_boundary_left_der.place(x=280, y=360)
lbl_boundary_right_der=Label(window, text="boundary_right_deriv")
lbl_boundary_right_der.place(x=130, y=400)
entry_boundary_right_der=Entry(window, width=50)
entry_boundary_right_der.place(x=280, y=400)
lbl_u0=Label(window, text="u0")
lbl_u0.place(x=130, y=440)
entry_u0=Entry(window, width=50)
entry_u0.place(x=280, y=440)
lbl_u0_der=Label(window, text="u0_der")
lbl_u0_der.place(x=130, y=480)
entry_u0_der=Entry(window, width=50)
entry_u0_der.place(x=280, y=480)
lbl_u_exact=Label(window, text="u_exact")
lbl_u_exact.place(x=130, y=520)
entry_u_exact=Entry(window, width=50)
entry_u_exact.place(x=280, y=520)
lbl_u_der_exact=Label(window, text="u_exact_deriv")
lbl_u_der_exact.place(x=130, y=560)
entry_u_der_exact=Entry(window, width=50)
entry_u_der_exact.place(x=280, y=560)

lbl_p=Label(window, text="p")
lbl_p.place(x=700, y=40)
entry_p=Entry(window)
entry_p.place(x=720, y=40)
lbl_n=Label(window, text="N")
lbl_n.place(x=700, y=80)
entry_n=Entry(window)
entry_n.place(x=720, y=80)
lbl_m=Label(window, text="M")
lbl_m.place(x=700, y=120)
entry_m=Entry(window)
entry_m.place(x=720, y=120)

entry_eps.insert(0, "1")
entry_b.insert(0, "1")
entry_c.insert(0, "1")
entry_l.insert(0, "1")
entry_t.insert(0, "1")
entry_f.insert(0, "(np.exp(1)-1)/1*(1-t-x*t)-t+np.exp(x)*t+np.exp(x)")
entry_boundary_left.insert(0, "1")
entry_boundary_right.insert(0, "np.exp(1)")
entry_boundary_left_der.insert(0, "0")
entry_boundary_right_der.insert(0, "0")
entry_u0.insert(0, "(np.exp(1)-1)/1*x+1")
entry_u0_der.insert(0, "(np.exp(1)-1)/10")
entry_u_exact.insert(0, "((np.exp(1)-1)/1*x+1)*(1-t)+np.exp(x)*t")
entry_u_der_exact.insert(0, "(np.exp(1)-1)/1*(1-t)+np.exp(x)*t")
entry_p.insert(0, "1")
entry_n.insert(0, "4")
entry_m.insert(0, "100")

"""entry_eps.insert(0, "1e-3")
entry_b.insert(0, "1")
entry_c.insert(0, "1")
entry_l.insert(0, "10")
entry_t.insert(0, "1")
entry_f.insert(0, "0")
entry_boundary_left.insert(0, "0")
entry_boundary_right.insert(0, "1")
entry_boundary_left_der.insert(0, "0")
entry_boundary_right_der.insert(0, "0")
entry_u0.insert(0, "x/10")
entry_u0_der.insert(0, "1/10")
entry_u_exact.insert(0, "0")
entry_u_der_exact.insert(0, "0")
entry_p.insert(0, "1")
entry_n.insert(0, "100")
entry_m.insert(0, "100")"""

def solve():
	theta=0.5
	eps=float(entry_eps.get())
	b=float(entry_b.get())
	c=float(entry_c.get())
	l=float(entry_l.get())
	T=float(entry_t.get())
	p=int(entry_p.get())
	N=int(entry_n.get())
	M=int(entry_m.get())
	h=l/N
	delta_t=T/M

	shape_functions = [Polynomial([0.5, -0.5]), Polynomial([0.5, 0.5])]
	shape_function_derivs = []
	stiffness_diffusion=np.zeros((N-1+(p-1)*N, N-1+(p-1)*N))
	stiffness_convection=np.zeros((N-1+(p-1)*N, N-1+(p-1)*N))
	gram=np.zeros((N-1+(p-1)*N, N-1+(p-1)*N))
	
	def f(t, x):
		return eval(entry_f.get(), {"np": np}, {"t": t, "x": x})

	def boundary_left(t):
		return eval(entry_boundary_left.get(), {"np": np}, {"t": t})

	def boundary_right(t):
		return eval(entry_boundary_right.get(), {"np": np}, {"t": t})

	def boundary_left_der(t):
		return eval(entry_boundary_left_der.get(), {"np": np}, {"t": t})

	def boundary_right_der(t):
		return eval(entry_boundary_right_der.get(), {"np": np}, {"t": t})

	def u0(x):
		return eval(entry_u0.get(), {"np": np}, {"x": x})

	def u0_der(x):
		return eval(entry_u0_der.get(), {"np": np}, {"x": x})

	def u_exact(t, x):
		return eval(entry_u_exact.get(), {"np": np}, {"t": t, "x": x})

	def u_der_exact(t, x):
		return eval(entry_u_der_exact.get(), {"np": np}, {"t": t, "x": x})

	def project_u0_onto_subspace():
		B=np.empty(N-1+(p-1)*N)
		for k in range(N):
			for i in range(p):
				if i == 0:
					if k > 0:
						shape_functions[1].domain=[(k-1)*h, k*h]
						shape_functions[0].domain=[k*h, (k+1)*h]
						B[k-1]=(integrate.quad(lambda x:\
						u0(x)*shape_functions[1](x)+u0_der(x)/h, (k-1)*h, k*h)[0]\
						+integrate.quad(lambda x:\
						u0(x)*shape_functions[0](x)-u0_der(x)/h, k*h, (k+1)*h)[0])
						if k == 1:
							B[0]-=u0(0)*(h/6-1/h)
						if k == N-1:
							B[N-2]-=u0(l)*(h/6-1/h)
				else:
					shape_functions[i+1].domain=[k*h, (k+1)*h]
					Ni_der=shape_functions[i+1].deriv()
					B[N-1+k*(p-1)+i-1]=integrate.quad(lambda x:\
					u0(x)*shape_functions[i+1](x)+u0_der(x)*Ni_der(x),\
					k*h, (k+1)*h)[0]
					if k == 0:
						if i == 1:
							B[N-1]+=u0(0)*h/(2*np.sqrt(6))
						if i == 2:
							B[N-1+1]-=u0(0)*h/(6*np.sqrt(10))
					if k == N-1:
						if i == 1:
							B[N-1+(N-1)*(p-1)]+=u0(l)*h/(2*np.sqrt(6))
						if i == 2:
							B[N-1+(N-1)*(p-1)+1]+=u0(l)*h/(6*np.sqrt(10))
		return np.linalg.solve(gram+stiffness_diffusion, B)

	def find_F(t):
		F=np.zeros(N-1+(p-1)*N)
		for k in range(N):
			domain=[k*h, (k+1)*h]
			for i in range(p):
				if i == 0:
					if k > 0:
						shape_functions[1].domain=[(k-1)*h, k*h]
						shape_functions[0].domain=domain
						F[k-1]=integrate.quad(lambda x:\
						f(t, x)*shape_functions[1](x), (k-1)*h, k*h)[0]\
						+integrate.quad(lambda x:\
						f(t, x)*shape_functions[0](x), k*h, (k+1)*h)[0]
				else:
					shape_functions[i+1].domain=domain
					F[N-1+k*(p-1)+i-1]=integrate.quad(lambda x:\
					f(t, x)*shape_functions[i+1](x), k*h, (k+1)*h)[0]
		return F

	def u_ti_approx(q, u_left, u_right, x):
		k=min(N-1, int(x/h))
		for i in range(p+1):
			shape_functions[i].domain=[k*h, (k+1)*h]
		res=(q[k-1] if k > 0 else u_left)*shape_functions[0](x)\
		+(q[k] if k < N-1 else u_right)*shape_functions[1](x)
		for i in range(2, p+1):
			res+=q[N-1+k*(p-1)+i-2]*shape_functions[i](x)
		return res

	# може бути не дуже точно у вузлах, треба indirect
	def u_der_approx(q, u_left, u_right, x):
		k=min(N-1, int(x/h))
		for i in range(p+1):
			shape_function_derivs[i].domain=[k*h, (k+1)*h]
		res=(q[k-1] if k > 0 else u_left)*shape_function_derivs[0](x)\
		+(q[k] if k < N-1 else u_right)*shape_function_derivs[1](x)
		for i in range(2, p+1):
			res+=q[N-1+k*(p-1)+i-2]*shape_function_derivs[i](x)
		return res

	def find_u_norm():
		u_norm=integrate.quad(lambda x: u_exact(T, x), 0, l)[0]/2
		for j in range(M):
			u_der_j=lambda x: u_der_exact(j/M*T, x)
			u_der_j1=lambda x: u_der_exact((j+1)/M*T, x)
			u_norm+=delta_t/3*integrate.quad(lambda x:\
			u_der_j(x)**2+u_der_j(x)*u_der_j1(x)+u_der_j1(x)**2, 0, l)[0]
		return np.sqrt(u_norm)

	for i in range(3, p + 2):
		shape_functions.append((Legendre.basis(i-1)-Legendre.basis(i-3))/np.sqrt(2*(2*i-3)))
	for shape_function in shape_functions:
		shape_function.domain=[0, h]
		shape_function_derivs.append(shape_function.deriv())

	for k in range(N):
		for i in range(p):
			if i == 0:
				if k > 0:
					stiffness_diffusion[k-1][k-1]=2/h
					if k < N-1:
						stiffness_diffusion[k-1][k]=-1/h
						stiffness_diffusion[k][k-1]=-1/h
			else:
				stiffness_diffusion[N-1+k*(p-1)+i-1][N-1+k*(p-1)+i-1]=2/h
		
		for i in range(p):
			if i == 0:
				if k > 0:
					if k < N-1:
						stiffness_convection[k-1][k]=b/2 #12
						stiffness_convection[k][k-1]=-b/2 #21
					if p >= 2:
						stiffness_convection[k-1][N-1+(k-1)*(p-1)]=b/np.sqrt(6) #23
						stiffness_convection[k-1][N-1+k*(p-1)]=-b/np.sqrt(6) #13
						stiffness_convection[N-1+(k-1)*(p-1)][k-1]=-b/np.sqrt(6) #32
						stiffness_convection[N-1+k*(p-1)][k-1]=b/np.sqrt(6) #31
			elif i <= p-2:
				stiffness_convection[N-1+k*(p-1)+i-1][N-1+k*(p-1)+i]=b/np.sqrt((2*i+3)*(2*i+1)) #i,i+1
				stiffness_convection[N-1+k*(p-1)+i][N-1+k*(p-1)+i-1]=-b/np.sqrt((2*i+3)*(2*i+1)) #i+1,i
				
		for i in range(p):
			if i == 0:
				if k > 0:
					gram[k-1][k-1]=2*h/3
					if k < N-1:
						gram[k-1][k]=h/6
						gram[k][k-1]=h/6
					if p >= 2:
						gram[k-1][N-1+(k-1)*(p-1)]=-h/(2*np.sqrt(6))
						gram[N-1+(k-1)*(p-1)][k-1]=-h/(2*np.sqrt(6))
						gram[k-1][N-1+k*(p-1)]=-h/(2*np.sqrt(6))
						gram[N-1+k*(p-1)][k-1]=-h/(2*np.sqrt(6))
					if p >= 3:
						gram[k-1][N-1+(k-1)*(p-1)+1]=-h/(6*np.sqrt(10))
						gram[N-1+(k-1)*(p-1)+1][k-1]=-h/(6*np.sqrt(10))
						gram[k-1][N-1+k*(p-1)+1]=h/(6*np.sqrt(10))
						gram[N-1+k*(p-1)+1][k-1]=h/(6*np.sqrt(10))
			else:
				gram[N-1+k*(p-1)+i-1][N-1+k*(p-1)+i-1]=h/((2*i+3)*(2*i-1))
				if i+3 <= p:
					gram[N-1+k*(p-1)+i-1][N-1+k*(p-1)+i+1]=-h/(2*(2*i+3)*np.sqrt((2*i+1)*(2*i+5)))
					gram[N-1+k*(p-1)+i+1][N-1+k*(p-1)+i-1]=-h/(2*(2*i+3)*np.sqrt((2*i+1)*(2*i+5)))
		

	A=eps*stiffness_diffusion+b*stiffness_convection+c*gram
	matr_inv=np.linalg.inv(gram+theta*delta_t*A)
	q=[project_u0_onto_subspace()]
	for j in range(M):
		print(j)
		t=(j+theta)/M*T
		F=find_F(t)-np.matmul(A, q[j])
		bnd_left_der=boundary_left_der(t)
		bnd_left=boundary_left(t)+delta_t*theta*bnd_left_der
		bnd_right_der=boundary_right_der(t)
		bnd_right=boundary_right(t)+delta_t*theta*bnd_right_der
		if N > 1:
			F[0]-=bnd_left*(-eps/h-b/2+c*h/6)+bnd_left_der*h/6 #21
			F[N-2]-=bnd_right*(-eps/h+b/2+c*h/6)+bnd_right_der*h/6 #12
		if p >= 2:
			F[N-1]-=bnd_left*(b/np.sqrt(6)-c*h/(2*np.sqrt(6)))\
			+bnd_left_der*(-h/(2*np.sqrt(6))) #31
			F[N-1+(N-1)*(p-1)]-=bnd_right*(-b/np.sqrt(6)-c*h/(2*np.sqrt(6)))\
			+bnd_right_der*(-h/(2*np.sqrt(6))) #32
		if p >= 3:
			F[N-1+1]-=bnd_left*c*h/(6*np.sqrt(10))\
			+bnd_left_der*h/(6*np.sqrt(10)) #41
			F[N-1+(N-1)*(p-1)+1]-=bnd_right*(-c*h/(6*np.sqrt(10)))\
			+bnd_right_der*(-h/(6*np.sqrt(10))) #42
		q.append(q[j]+delta_t*np.matmul(matr_inv, F))
	
	print("calculated")
	def u_approx(t, x):
		j_real = t / T * M
		j = min(M - 1, int(j_real))
		qt = ((j + 1) - j_real) * q[j] + (j_real - j) * q[j + 1]
		return u_ti_approx(qt, boundary_left(t), boundary_right(t), x)
	
	class UpdateDist:
		def __init__(self, ax):
			self.line_exact, = ax.plot([], [])
			self.line_approx, = ax.plot([], [])
			self.x = np.linspace(0, l, p * N)
			self.ax = ax

			# Set up plot parameters
			self.ax.set_xlim(0, l)
			self.ax.set_ylim(-2, 2)

		def __call__(self, t):
			self.line_exact.set_data(self.x, u_exact(t, self.x))
			self.line_approx.set_data(self.x, [u_approx(t, x) for x in self.x])
			return self.line_exact, self.line_approx

	fig, ax = plt.subplots()
	ud = UpdateDist(ax)
	anim = FuncAnimation(fig, ud, frames=np.linspace(0, T, 100), interval=100, blit=True)
	plt.show()
	
	u_norm=integrate.quad(lambda x:\
	u_approx(T, x)**2,\
	0, l, vec_func=False)[0]/2
	for j in range(M):
		t_j=j/M*T
		t_j1=(j+1)/M*T
		u_der_j=lambda x: u_der_approx(q[j], boundary_left(t_j),\
		boundary_right(t_j), x)
		u_der_j1=lambda x: u_der_approx(q[j+1], boundary_left(t_j1),\
		boundary_right(t_j1), x)
		u_norm+=delta_t/3*integrate.quad(lambda x:\
		u_der_j(x)**2+u_der_j(x)*u_der_j1(x)+u_der_j1(x)**2, k*h, (k+1)*h,\
		vec_func=False)[0]
	u_norm=np.sqrt(u_norm)
	print(np.sqrt(integrate.quad(lambda x:\
	u_approx(T, x)**2,\
	0, l, vec_func=False)[0]/2))
	print("u_norm =", u_norm)
	error=integrate.quad(lambda x:\
	(u_approx(T, x)-u_exact(T, x))**2,\
	0, l, vec_func=False)[0]/2
	for j in range(M):
		t_j=j/M*T
		t_j1=(j+1)/M*T
		e_der_j=lambda x: u_der_approx(q[j], boundary_left(t_j),\
		boundary_right(t_j), x)-u_der_exact(t_j, x)
		e_der_j1=lambda x: u_der_approx(q[j+1], boundary_left(t_j1),\
		boundary_right(t_j1), x)-u_der_exact(t_j1, x)
		error+=delta_t/3*integrate.quad(lambda x:\
		e_der_j(x)**2+e_der_j(x)*e_der_j1(x)+e_der_j1(x)**2, k*h, (k+1)*h,\
		vec_func=False)[0]
	error=np.sqrt(error)
	print(np.sqrt(integrate.quad(lambda x:\
	(u_approx(T, x)-u_exact(T, x))**2,\
	0, l, vec_func=False)[0]/2))
	print("errors\n", error, error/1.5127001757295218)

#matplotlib.use('GTK4Agg')
btn=Button(window, text="Розв'язати", command=solve)
btn.place(x=750, y=300)
window.title('Розв\'язування задачі дифузії-адвекції-реакції')
window.geometry("1000x700+10+10")
window.mainloop()
