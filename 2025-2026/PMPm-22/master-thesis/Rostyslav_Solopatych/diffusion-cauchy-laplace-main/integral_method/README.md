# Integral-equation method for the Cauchy problem on a doubly-connected domain

This module implements a Nyström boundary-integral solver for the Cauchy
problem of the Laplace equation on the doubly-connected planar domain
$D = D_1 \setminus \overline{D_0}$ with outer boundary $\Gamma_1$ and inner
boundary $\Gamma_0$.

Given Cauchy data $(f_1, f_2)$ on the _accessible_ outer boundary
$\Gamma_1$, the solver reconstructs the harmonic function $u$ in $D$ — and
in particular its trace $u|_{\Gamma_0}$ and normal derivative
$\partial u/\partial\nu|_{\Gamma_0}$ on the inaccessible inner boundary
$\Gamma_0$.

The method follows the single-layer ansatz of the reference paper
(equations 2.1–2.11 are referenced throughout the source). This README
documents the parts of the implementation that depend on subtle
conventions: orientations and normals, the periodic log-singular
quadrature, the Tikhonov / L-curve regularisation, the dataset bridge,
and the way interior points are evaluated.

---

## 1. Geometry and conventions

### 1.1 Boundary parametrisations

Both curves are sampled on the uniform parameter grid

$$
t_j = \frac{j\pi}{M}, \qquad j = 0, 1, \dots, 2M-1,
$$

so each curve has $2M$ nodes. The reference geometry used throughout the
codebase is

$$
\Gamma_1: \quad x_1(t) = (1.3\cos t,\; \sin t),
$$

$$
\Gamma_0: \quad x_0(t) = \bigl(0.5\cos t,\; 0.4\sin t - 0.3\sin^2 t\bigr).
$$

Both curves are traversed counter-clockwise. See
[boundary.py](boundary.py).

### 1.2 Outward normal of $D$

The kernels of §2 use **the outward unit normal of the solution domain
$D$** at every target point — not the outward normal of $D_1$ or $D_0$
individually. For a CCW curve the right-hand normal of the tangent is
the outward normal of its enclosed region, so

$$
\nu_1(t) = \frac{(x'_{1,2}(t),\; -x'_{1,1}(t))}{\lVert x'_1(t)\rVert},
\qquad
\nu_0(t) = -\frac{(x'_{0,2}(t),\; -x'_{0,1}(t))}{\lVert x'_0(t)\rVert}.
$$

Note the sign flip on $\Gamma_0$: outward of $D$ on the inner boundary
points _into_ the hole $D_0$. This convention is enforced in
[`_sample_curve`](boundary.py) by the line
`normal = rhs_normal if is_outer else -rhs_normal`.

The same sign convention is used by the dataset generator (only
$\Gamma_1$ has Neumann data, and there both pipelines use the outward
ellipse normal — they agree exactly).

---

## 2. Single-layer formulation

We seek $u$ as a sum of single-layer potentials supported on both
boundary components,

$$
u(x) = \frac{1}{2\pi}\int_{\Gamma_0}\mu_0(y)\,\ln\frac{1}{|x-y|}\,ds(y)
     + \frac{1}{2\pi}\int_{\Gamma_1}\mu_1(y)\,\ln\frac{1}{|x-y|}\,ds(y),
\qquad x\in D.
\tag{2.1}
$$

Writing the _scaled_ densities $\psi_i(t) = \mu_i(x_i(t))\,|x_i'(t)|$
absorbs the line element $ds$ into the unknown, so the parametric
integrals all become $\int_0^{2\pi}(\cdot)\,d\tau$. The unknowns solved
for in the linear system are these scaled densities $\psi_0,\psi_1$.

Taking the trace and the normal derivative on $\Gamma_1$ gives the
boundary integral equations

$$
\int_{0}^{2\pi}H_{01}(t,\tau)\,\psi_0(\tau)\,d\tau
+ \int_{0}^{2\pi}H_{11}(t,\tau)\,\psi_1(\tau)\,d\tau
= f_1(t),
\tag{2.4}
$$

$$
\int_{0}^{2\pi}K_{01}(t,\tau)\,\psi_0(\tau)\,d\tau
+ \int_{0}^{2\pi}K_{11}(t,\tau)\,\psi_1(\tau)\,d\tau
+ \frac{\psi_1(t)}{2|x_1'(t)|}
= f_2(t),
\tag{2.5}
$$

with kernels

$$
H_{ij}(t,\tau) = \frac{1}{2\pi}\ln\frac{1}{|x_i(t) - x_j(\tau)|},
\qquad
K_{ij}(t,\tau) = \frac{1}{2\pi}\,
\frac{(x_j(\tau) - x_i(t))\cdot \nu(x_i(t))}{|x_i(t) - x_j(\tau)|^2}.
$$

The first index of $H_{ij}, K_{ij}$ is the _target_ curve and the second
is the _source_; $\nu$ is the outward-of-$D$ normal at the target.

Both off-curve kernels ($i\neq j$) are smooth. The self-kernels $H_{11}$
and $K_{11}$ are singular only on the diagonal $t=\tau$ and admit
explicit closed-form limits (see §3).

The $\psi_1(t)/(2|x_1'(t)|)$ term in (2.5) is the **jump relation** of
the normal derivative of the single layer; it produces the $+\tfrac12 I$
diagonal that makes the second-row block well-conditioned.

---

## 3. Nyström discretisation

### 3.1 Singular splitting of $H_{ii}$

Direct trapezoidal quadrature of $H_{ii}$ would converge slowly because
of the diagonal $\ln|x_i(t)-x_i(\tau)|$ singularity. Following Kress we
split

$$
H_{ii}(t,\tau)
= \tfrac{1}{2}\,\ln\!\left(\frac{4}{e}\sin^2\frac{t-\tau}{2}\right)
+ \widetilde H_{ii}(t,\tau),
$$

where the first term is an **explicit periodic log-singular kernel** and
the remainder $\widetilde H_{ii}$ is smooth, with diagonal limit

$$
\widetilde H_{ii}(t,t) = \tfrac12\,\ln\!\frac{1}{e\,|x_i'(t)|^2}.
$$

Off-diagonal,

$$
\widetilde H_{ii}(t,\tau)
= \tfrac{1}{2}\Bigl[\ln(4/e) + \ln\sin^2\tfrac{t-\tau}{2}
                   - \ln|x_i(t)-x_i(\tau)|^2\Bigr].
$$

Both pieces are implemented in [`kernels.py`](kernels.py) (`_self_log_smooth`).

### 3.2 Diagonal limit of $K_{ii}$

Although $K_{ii}$ looks $1/r^2$-singular, the numerator
$(x_i(\tau)-x_i(t))\cdot\nu_i(t)$ vanishes to second order on the
diagonal. The resulting limit is

$$
K_{ii}(t,t)
= \frac{x_i''(t)\cdot\nu_i(t)}{2\,|x_i'(t)|^2}.
$$

This is filled into the diagonal of the kernel matrix in
`_self_double_layer`.

### 3.3 Trapezoidal rule for smooth periodic kernels

For all smooth kernels (the off-curve $H_{ij}, K_{ij}$ with $i\neq j$,
the smooth remainder $\widetilde H_{ii}$, and the regularised $K_{ii}$
with its diagonal limit) we apply the periodic trapezoidal rule

$$
\frac{1}{2\pi}\int_0^{2\pi}f(\tau)\,d\tau
\approx \frac{1}{2M}\sum_{k=0}^{2M-1} f(t_k),
$$

which is **spectrally accurate** for analytic periodic integrands.

### 3.4 Martensen / Kress weights for the log-singular part

The remaining log-singular integrand needs a special quadrature. We
use the Martensen weights ([`quadrature.py`](quadrature.py)),

$$
\frac{1}{2\pi}\int_0^{2\pi}
\ln\!\left(\frac{4}{e}\sin^2\!\frac{t-\tau}{2}\right) g(\tau)\,d\tau
\approx \sum_{k=0}^{2M-1} R_k(t)\,g(t_k),
$$

with the closed-form weights

$$
R_k(t)
= -\frac{1}{2M}
- \frac{1}{M}\sum_{m=1}^{M-1}\frac{\cos m(t-t_k)}{m}
- \frac{\cos M(t-t_k)}{2M^2}.
$$

These integrate trigonometric polynomials of degree $<2M$ exactly and,
combined with the splitting of §3.1, give exponential convergence for
analytic boundaries.

### 3.5 The discrete linear system

Putting everything together, the unknowns
$(\psi_{0,j}, \psi_{1,j})_{j=0}^{2M-1}$ satisfy the $4M\times 4M$ system
implemented in [`system.py`](system.py):

$$
\boxed{\;
\begin{aligned}
&\frac{1}{2M}\sum_{j} \psi_{0,j}\,H_{01}(t_i,t_j)
+ \sum_{j}\psi_{1,j}\!\left[\frac{\widetilde H_{11}(t_i,t_j)}{2M}
                              - \tfrac12 R_j(t_i)\right]
= f_1(t_i),\\[4pt]
&\frac{1}{2M}\sum_{j}\psi_{0,j}\,K_{01}(t_i,t_j)
+ \frac{1}{2M}\sum_{j}\psi_{1,j}\,K_{11}(t_i,t_j)
+ \frac{\psi_{1,i}}{2|x_1'(t_i)|}
= f_2(t_i).
\end{aligned}\;}
\tag{2.11}
$$

In block-matrix form,

$$
\underbrace{
\begin{pmatrix}
\frac{1}{2M}H_{01} & \frac{1}{2M}\widetilde H_{11} - \tfrac12 R\\[4pt]
\frac{1}{2M}K_{01} & \frac{1}{2M}K_{11} + \mathrm{diag}\!\left(\frac{1}{2|x_1'|}\right)
\end{pmatrix}}_{A}
\begin{pmatrix}\psi_0\\ \psi_1\end{pmatrix}
=
\begin{pmatrix}f_1\\ f_2\end{pmatrix}.
$$

---

## 4. Regularisation: Tikhonov + L-curve

### 4.1 Why regularise

The Cauchy problem for the Laplacian is **severely ill-posed**: small
perturbations of the boundary data $(f_1, f_2)$ — measurement noise,
discretisation error, or even floating-point error — can produce
arbitrarily large changes in the densities and hence in the
reconstructed $u|_{\Gamma_0}$. The discrete operator $A$ inherits this
ill-posedness as **exponentially decaying singular values**: with $M=32$
nodes on the reference geometry $\mathrm{cond}(A)$ is already
$10^{12}$–$10^{14}$, so naive solves blow up.

### 4.2 Tikhonov solution via SVD

We replace $Ax=b$ by the regularised normal equation

$$
x_\lambda
= \arg\min_{x}\;\bigl\lVert Ax - b\bigr\rVert_2^2
            + \lambda\,\lVert x\rVert_2^2.
$$

Using the thin SVD $A = U\Sigma V^\top$ with $\Sigma=\mathrm{diag}(\sigma_i)$,

$$
x_\lambda = V\,\mathrm{diag}\!\left(\frac{\sigma_i}{\sigma_i^2+\lambda}\right) U^\top b,
\qquad
f_i(\lambda) = \frac{\sigma_i^2}{\sigma_i^2+\lambda}.
$$

The numerator $\sigma_i$ instead of $1/\sigma_i$ in the formula above is
the numerically stable form: it stays finite as $\sigma_i\to 0$, since
the filter factor $f_i$ kills small singular components.

For the L-curve we need both norms in closed form:

$$
\lVert x_\lambda\rVert_2^2
= \sum_i \left(\frac{\sigma_i}{\sigma_i^2+\lambda}\right)^{\!2}\!(U^\top b)_i^2,
$$

$$
\lVert A x_\lambda - b\rVert_2^2
= \sum_i \left(\frac{\lambda}{\sigma_i^2+\lambda}\right)^{\!2}\!(U^\top b)_i^2
  + \bigl\lVert b - U U^\top b\bigr\rVert_2^2.
$$

The last "out-of-range" term is independent of $\lambda$ and vanishes
for square invertible $A$; we keep it for generality (and for square
$A$ it is numerically zero up to round-off).

### 4.3 L-curve corner selection

Plotting $(\rho(\lambda), \eta(\lambda)) = \bigl(\log\lVert Ax\_\lambda

- b\rVert,\, \log\lVert x\_\lambda\rVert\bigr)$ on log-log axes typically
  produces an "L" shape with three regimes:

- **Vertical leg** ($\lambda$ too small): the residual is small but
  $\lVert x_\lambda\rVert$ explodes — noise has been amplified through
  the small singular values.
- **Horizontal leg** ($\lambda$ too large): the residual grows because
  the solution is over-smoothed; $\lVert x_\lambda\rVert$ is small.
- **Corner**: the optimal trade-off, where the residual just begins to
  rise sharply with decreasing $\lambda$.

We pick $\lambda^*$ at the point of **maximum (signed) curvature** of
the parametric log-log curve,

$$
\kappa(\lambda) =
\frac{\rho'\eta'' - \rho''\eta'}{\bigl(\rho'^2 + \eta'^2\bigr)^{3/2}},
$$

with derivatives w.r.t. the (uniform) log-$\lambda$ parameter and
endpoints excluded. See `_lcurve_corner` in
[`tikhonov.py`](tikhonov.py). The default search grid is
`np.geomspace(1e-14, 1e2, 200)`.

This is Hansen's standard heuristic; it is parameter-free and
performs well on this problem because $A$'s singular values decay
smoothly, producing a well-defined corner.

---

## 5. Reconstruction on $\Gamma_0$

Once $\psi_0, \psi_1$ are known, the trace and normal derivative on
$\Gamma_0$ follow from the same single-layer representation, taking the
appropriate jump relations for the inner curve:

$$
u(x_0(t_i))
= \sum_j \left[\frac{\widetilde H_{00}(t_i,t_j)}{2M} - \tfrac12 R_j(t_i)\right]\psi_{0,j}
+ \sum_j \frac{H_{10}(t_i,t_j)}{2M}\,\psi_{1,j},
$$

$$
\frac{\partial u}{\partial \nu}(x_0(t_i))
= -\frac{\psi_{0,i}}{2|x_0'(t_i)|}
+ \frac{1}{2M}\sum_j K_{00}(t_i,t_j)\,\psi_{0,j}
+ \frac{1}{2M}\sum_j K_{10}(t_i,t_j)\,\psi_{1,j}.
$$

The sign of the jump term flips compared with $\Gamma_1$ (the $-\psi_0/(2|x_0'|)$
on $\Gamma_0$ versus $+\psi_1/(2|x_1'|)$ on $\Gamma_1$) because the
outward-of-$D$ normal points _into_ $D_0$ — the limit is taken from
inside $D$, i.e. from outside $D_0$. See `reconstruct_on_inner` in
[`reconstruct.py`](reconstruct.py).

---

## 6. Reconstruction at interior points

### 6.1 The integrals collapse to a smooth periodic sum

For a strictly interior point $x \in D$ (off both curves), the kernel
$\ln(1/|x-y|)$ is **smooth and $2\pi$-periodic** in the source parameter
$\tau$, so no special quadrature is required. Direct application of the
periodic trapezoidal rule to (2.1) gives

$$
u(x)
\approx \frac{1}{2M}\sum_{j=0}^{2M-1}\psi_{0,j}\,
        \ln\frac{1}{|x - x_0(t_j)|}
\;+\; \frac{1}{2M}\sum_{j=0}^{2M-1}\psi_{1,j}\,
        \ln\frac{1}{|x - x_1(t_j)|}.
$$

The factor $1/(2M)$ combines the kernel prefactor $1/(2\pi)$ with the
trapezoidal weight $\pi/M$:
$\tfrac{1}{2\pi}\cdot\tfrac{\pi}{M} = \tfrac{1}{2M}$. The line-element
$|x_i'(t_j)|$ is _not_ multiplied in here because it is already absorbed
into the scaled density $\psi$.

### 6.2 Implementation

`evaluate_interior(points, gamma0, gamma1, psi0, psi1, M)` in
[`reconstruct.py`](reconstruct.py) is a pure matrix-vector product: it
builds the $(P, 2M)$ kernel matrix
$H_{ij} = -\tfrac12\ln|x_i - x_1(t_j)|^2$ and returns
$\frac{1}{2M}(H_0\psi_0 + H_1\psi_1)$.

`evaluate_on_grid` wraps this for a pixel raster, evaluating only at
pixels where a user-supplied `fill_mask` is `True` — typically
`gmask | bmask`, i.e. the interior plus the $\Gamma_1$ boundary band
(the same pixels covered by the dataset's `u` field).

### 6.3 Accuracy caveat: near-boundary "close evaluation" loss

The trapezoidal rule on the smooth periodic integrand converges
**exponentially** _only_ when $x$ is sufficiently far from the source
curve. As $x \to \Gamma_0$ or $x \to \Gamma_1$, the kernel
$\ln|x - y|$ becomes nearly singular and an increasing number of nodes
is needed to resolve the rapid variation across the curve. For pixels
within a few nodes of the boundary the error degrades from
spectral to algebraic. This is the classical **close-evaluation
problem** in boundary integral equations.

In this code we do **not** apply a near-singular correction (e.g.
QBX or kernel-split close-evaluation schemes); the assumption is that
$M$ is chosen large enough relative to the pixel resolution that the
error within a single pixel of $\Gamma_1$ is acceptable. If this is
ever violated, the visible symptom is a thin, oscillatory layer of
error along the boundary in the reconstructed image.

---

## 7. Synthetic data and the dataset bridge

### 7.1 Synthetic Cauchy data ([`synthetic.py`](synthetic.py))

Ground-truth harmonic fields are taken as real parts of complex
polynomials, $u(x,y) = \sum_n a_n\,\mathrm{Re}(z^n)$, $z=x+iy$. The
trace on $\Gamma_1$ supplies $f_1$, and the normal derivative

$$
f_2(t) = (\nabla u)(x_1(t)) \cdot \nu_1(t)
$$

is computed from the closed-form gradient. Optional Gaussian noise of
prescribed percentage of the discrete $L^2$-norm is added to both
components — used for stability tests in §4.

### 7.2 Dataset bridge ([`from_dataset.py`](from_dataset.py))

The DDPM dataset stores Cauchy data on **boundary pixels** of $\Gamma_1$
on a Cartesian raster $[-1.5,1.5]^2$ at resolution `pixel_res`. To
feed it into the integral solver we map each boundary pixel
$(x_b, y_b)$ to its ellipse parameter

$$
t_b = \mathrm{atan2}\!\bigl(y_b/b,\; x_b/a\bigr) \bmod 2\pi,
\qquad a=1.3,\ b=1,
$$

then perform periodic 1-D linear interpolation onto the parametric
nodes $t_j = j\pi/M$. The interpolated $(f_1, f_2)$ feed
`assemble_system → tikhonov_lcurve → evaluate_on_grid` to produce a
pixel image of $u$ matching the dataset's layout.

---

## 8. File map

| File                                           | Contents                                                               |
| ---------------------------------------------- | ---------------------------------------------------------------------- |
| [boundary.py](boundary.py)                     | Curve sampling, derivatives, outward-of-$D$ normals                    |
| [kernels.py](kernels.py)                       | $H_{ij}$, $K_{ij}$ blocks with singular splittings and diagonal limits |
| [quadrature.py](quadrature.py)                 | Martensen / Kress weights $R_k(t_i)$                                   |
| [system.py](system.py)                         | Block assembly of the $(4M\times 4M)$ linear system                    |
| [tikhonov.py](tikhonov.py)                     | SVD-based Tikhonov solve and L-curve corner                            |
| [reconstruct.py](reconstruct.py)               | Trace/normal derivative on $\Gamma_0$ and interior evaluation          |
| [synthetic.py](synthetic.py)                   | Harmonic-polynomial ground-truth generator                             |
| [from_dataset.py](from_dataset.py)             | Bridge from pixel-grid Cauchy data to parametric nodes                 |
| [run_reconstruction.py](run_reconstruction.py) | CLI driver: synthetic data → reconstruction → plots                    |
| [test\_\*.py](.)                               | Unit tests for boundary, quadrature, system, reconstruction            |

---

## 9. Reference equations (for paper reuse)

For convenience, the central equations that should appear verbatim in
any write-up:

**Single-layer ansatz**

$$
u(x) = \sum_{i=0}^{1}\frac{1}{2\pi}\int_{\Gamma_i}\mu_i(y)\,\ln\frac{1}{|x-y|}\,ds(y).
$$

**Boundary integral equations on $\Gamma_1$**

$$
\sum_{j=0}^{1}\!\int_{\Gamma_j}\!H_{1j}(x,y)\mu_j(y)\,ds(y) = f_1(x),
\quad
\sum_{j=0}^{1}\!\int_{\Gamma_j}\!K_{1j}(x,y)\mu_j(y)\,ds(y) + \tfrac12\mu_1(x) = f_2(x).
$$

**Nyström system**

$$
\begin{pmatrix}
\frac{1}{2M}H_{01} & \frac{1}{2M}\widetilde H_{11} - \tfrac12 R\\[4pt]
\frac{1}{2M}K_{01} & \frac{1}{2M}K_{11} + \mathrm{diag}\!\bigl(\tfrac{1}{2|x_1'|}\bigr)
\end{pmatrix}
\!\begin{pmatrix}\psi_0\\ \psi_1\end{pmatrix}
=\!\begin{pmatrix}f_1\\ f_2\end{pmatrix}.
$$

**Tikhonov / L-curve**

$$
x_\lambda = V\,\mathrm{diag}\!\left(\frac{\sigma_i}{\sigma_i^2+\lambda}\right) U^\top b,
\qquad
\lambda^* = \arg\max_\lambda \kappa(\lambda).
$$

**Interior evaluation**

$$
u(x) \approx \frac{1}{2M}\sum_{j=0}^{2M-1}\!
\Bigl[\psi_{0,j}\,\ln\tfrac{1}{|x-x_0(t_j)|}
    + \psi_{1,j}\,\ln\tfrac{1}{|x-x_1(t_j)|}\Bigr].
$$
