# PINN (Physics-Informed Neural Network)
Code for solving various PDEs with PINN.

Solutions are compared to exact solution or FEM solution.

Here is a list of equations that are solved:

1. File 2DRobin.ipynb solves Laplace's 2D equation:

<!-- $$
\bigtriangleup u = 0, \quad x \in \Omega = (0, 1) \times (0, 1),
$$ --> 

<div align="center"><img style="background: white;" src="./svg/S9M8GPB77l.svg"></div>

<!-- $$
u = 0\text{ on }\Gamma_1, 
$$ --> 

<div align="center"><img style="background: white;" src="./svg/id6IIGqanU.svg"></div>
<!-- $$
\frac{\partial u}{\partial n} = u\text{ on }\Gamma_2, 
$$ --> 

<div align="center"><img style="background: white;" src="./svg/re3F4170Fa.svg"></div>
<!-- $$
u = 1\text{ on }\Gamma_3, 
$$ --> 

<div align="center"><img style="background: white;" src="./svg/VctwuXLdEt.svg"></div>
<!-- $$
\frac{\partial u}{\partial n} = 0\text{ on }\Gamma_4.
$$ --> 

<div align="center"><img style="background: white;" src="./svg/uqpDWwthWc.svg"></div>

<p align="center">
    <img src="./imgs/SquareRegion.png" alt="Image of region with boundaries" style="width:40%; border:0;">
</p>

2. File 1DNonStationaryHeatEquation.ipynb solves non-stationary 1D heat equation:

<!-- $$
\frac{\partial u}{\partial t} = \frac{\partial^2 u}{\partial x^2}, \quad (x,t) \in \Omega = (0, 1)^2 \times (0, 0.2),
$$ --> 

<div align="center"><img style="background: white;" src="./svg/ORlKv5MBjw.svg"></div>

<!-- $$
u(0, t) = 0,
$$ --> 

<div align="center"><img style="background: white;" src="./svg/AcBI6saU4V.svg"></div>

<!-- $$
u(1, t) = 0,
$$ --> 

<div align="center"><img style="background: white;" src="./svg/9Fumxo3hJh.svg"></div>

<!-- $$
u(x, 0) = \sin(\pi x) - \sin(2 \pi x) + \sin(3 \pi x).
$$ --> 

<div align="center"><img style="background: white;" src="./svg/JMre20fsuI.svg"></div>

3. File 2DNonStationaryHeatEquation.ipynb solves non-stationary 2D heat equation:

<!-- $$
\frac{\partial u}{\partial t} = 0.01\bigtriangleup u, \quad (x,t) \in \Omega = (0, 1)^2\times (0, 2),
$$ --> 

<div align="center"><img style="background: white;" src="./svg/pU7zKnk2yP.svg"></div>

<!-- $$
u(0, x_2, t) = u(x_{1 max},x_2, t) = u(x_1, 0, t) = u(x_1, x_2, 0) = 0,
$$ --> 

<div align="center"><img style="background: white;" src="./svg/4j1CPwa9oW.svg"></div>

<!-- $$
u(x_1, x_{2 max}, t) = 1.
$$ --> 

<div align="center"><img style="background: white;" src="./svg/MyaPF7Bwkg.svg"></div>

4. File 1DStationaryHeatEquationRing.ipynb solves 2D Laplace's equation with Dirichlet conditions in double-connected region

<!-- $$
\Gamma_1=\{x_1(\varphi)=(2cos(\varphi), 2sin(\varphi)), \varphi \in [0, 2\pi]\},
$$ --> 

<div align="center"><img style="background: white;" src="./svg/tYcIle2BJv.svg"></div>
<!-- $$
\Gamma_2=\{x_2(\varphi)=(5cos(\varphi), 5sin(\varphi)), \varphi \in [0, 2\pi]\},
$$ --> 

<div align="center"><img style="background: white;" src="./svg/M0LD0SeN2i.svg"></div>
<!-- $$
\bigtriangleup u=0\text{ in } D \text{(a region between }\Gamma_1\text{ and }\Gamma_2),
$$ --> 

<div align="center"><img style="background: white;" src="./svg/x6MfE1D1Nj.svg"></div>
<!-- $$
u=x\text{ on }\Gamma_1,
$$ --> 

<div align="center"><img style="background: white;" src="./svg/EuvspOXJVj.svg"></div>
<!-- $$
u=0\text{ on }\Gamma_2.
$$ --> 

<div align="center"><img style="background: white;" src="./svg/lolWWx5i5E.svg"></div>

<p align="center">
    <img src="./imgs/Double_connected_domain.png" alt="Image of region with boundaries" style="width:80%; border:0;">
</p>