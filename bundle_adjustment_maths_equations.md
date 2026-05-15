# Bundle Adjustment for Sonar Geometry

## Assumptions.

1. NED reference frame

2. Upper idxs: x$^s$ means source frame, x$^t$ means target frame, x$^g$ means global frame,  

3. 

## Projection/Reprojection model

**1. FLS points to 3D points**

Pixel of fls image: 
$$
p^s_i = [r^s, \theta^s]^T
$$
Each pixel has assign estimated elevation angle $\phi_i$.

This pixel represent point in 3D space, in body reference frame:

$$ 
P^s_i = \Pi^{-1}(r, \theta, \phi)=
\begin{bmatrix} 
r^s \cos\phi^s \cos\theta^s \\ 
r^s \cos\phi^s \sin\theta^s \\ 
r^s \sin\phi^s 
\end{bmatrix} =
\begin{bmatrix} x^s \\ y^s \\ z^s \end{bmatrix} 
$$

**Note:** $P_i$ is in body reference frame, not global frame!

**2. Reprojection**

Now point in source pose reference frame is reprojetcted to target pose frame:

**Source frame** -> **Global frame** -> **Target frame** 

$$ 
P^t_i = 
\mathbf{T}_t^{-1} \mathbf{T}_s P^s = 
\begin{bmatrix} x^t \\ y^t \\ z^t \end{bmatrix} 
$$

where:
$$ \mathbf{T}_t $$ is transform (Lie group SE3 object) from target pose frame to global frame. 
$$ \mathbf{T}_s $$ is transform (Lie group SE3 object) from source pose frame to global frame. 

Lie gruop object $SE(3)$ consist of: 
- Rotation component ( $\mathbf{R} \in SO(3)$)
- Translation component ( $\mathbf{t} \in \mathbb{R}^3$ )

**3. 3D points to FLS points**

Now project points to forward looking sonar pixels. 

$$ 
p^t = \Pi(i, y, z)=
\begin{bmatrix} r^t \\ \theta^t \end{bmatrix} = 
\begin{bmatrix} 
\sqrt{{x^t}^2 + {y^t}^2 + {z^t}^2} \\ 
\arctan(\frac{y^t}{x^t}) \end{bmatrix} 
$$

**4. Reprojection error**

Now let's define reprojection error:

$ p^t_{0} $ - reprojected points with opses from 0 iteration of optimization (before optimization)
$ p^t_{i} $ - reprojeted poses wit i-th iteration of optimization

$$
e = p^t_{0} + \delta -  p^t_{i}
$$

where: $\delta$ - correction to optic flow, from GRU module

when we mark constant (for specific optimization process) reprojection baseline: 
$$
e_{ref} = p^t_{0} + \delta
$$
Then we can present reproejction error as: 

$$
e(r, \theta, \phi) = e_{ref} - \Pi(\mathbf{T}_t^{-1} \mathbf{T}_s\Pi^{-1}(r, \theta, \phi))
$$

## Jacobians

Optimize parameters: 
- $ \mathbf{T}_t $
- $ \mathbf{T}_s $
- $ \phi^s $

So we need to define 3 jacobians: 

$$
J_{\mathbf{T}_t} = \frac{\partial e}{\partial \mathbf{T}_t},  
J_{\mathbf{T}_s} = \frac{\partial e}{\partial \mathbf{T}_s},  
J_{\phi^s} = \frac{\partial e}{\partial \phi^s}
$$


To do this we use **Chain rule**:

$$ 
\frac{\partial e}{\partial (x)} = 
\frac{\partial e}{\partial p^t} 
\frac{\partial p^t}{\partial P^t} 
\frac{\partial P^t}{\partial (x)} 
$$


Note: 
If 
$ e = p^t_{0} + \delta -  p^t_{i} $, then $ \frac{\partial e}{\partial p^t_{i}} = - 1$

So chain rule can be siimplify to:


$$ 
\frac{\partial e}{\partial (x)} = 
-\frac{\partial p^t}{\partial P^t} 
\frac{\partial P^t}{\partial (x)} 
$$


First, we need the derivative of the projection function $\Pi$ with respect to the 3D point in the target frame $P^t = [x^t, y^t, z^t]^T$. Let's call it $\mathbf{J}_\Pi$.

$$
\mathbf{J}_\Pi = 
\frac{\partial p^t}{\partial P^t} = 
\begin{bmatrix} 

\frac{\partial r^t}{\partial x^t} & 
\frac{\partial r^t}{\partial y^t} & 
\frac{\partial r^t}{\partial z^t} \\ 

\frac{\partial \theta^t}{\partial x^t} & 
\frac{\partial \theta^t}{\partial y^t} & 
\frac{\partial \theta^t}{\partial z^t} 

\end{bmatrix}
$$

Using equations from section 3 (NED frame):
**SPRAWDZIĆ!!!!!!**
$$
\mathbf{J}_\Pi = 
\begin{bmatrix} 
\frac{x^t}{r^t} & \frac{y^t}{r^t} & \frac{z^t}{r^t} \\ 
\frac{-y^t}{{x^t}^2 + {y^t}^2} & \frac{x^t}{{x^t}^2 + {y^t}^2} & 0 
\end{bmatrix}

$$

(Note: $r^t = \sqrt{{x^t}^2 + {y^t}^2 + {z^t}^2}$).

**Chat disclaimer for explanation belowed equations**:

Absolutnie, nie ma w tym nic dziwnego! Różniczkowanie na rozmaitościach (czyli algebrze Liego $SE(3)$) to jeden z najtrudniejszych koncepcyjnie elementów w robotyce. 
  
Standardowe zasady z analizy matematycznej przestają tu działać wprost, co potrafi przyprawić o zawrót głowy.Rozłóżmy to na czynniki pierwsze, powoli i po polsku. Zanim przejdziemy do punktów 2, 3 i 4, musimy zrozumieć "magię" algebry Liego, która się tam dzieje.Wstęp: Dlaczego nie liczymy normalnej pochodnej?
  
Gdybyś miał zwykłą liczbę $x$, pochodną liczyłbyś dodając do niej malutkie $\delta x$: $x \leftarrow x + \delta x$.
  
Ale $\mathbf{T}$ to macierz $4\times4$, która musi spełniać ścisłe reguły (musi być ortogonalna, wyznacznik rotacji = 1 itd.). Gdybyś po prostu dodał do niej malutką macierz $\delta \mathbf{T}$, przestałaby być poprawną transformacją 3D.
  
Dlatego zamiast dodawać, w algebrze Liego mnożymy macierz przez "malutką transformację". 
  
Tą malutką transformacją jest $\exp(\delta \boldsymbol{\xi}^\wedge)$.
$\delta \boldsymbol{\xi} = [\mathbf{v}, \boldsymbol{\omega}]^T$ to wektor 6-elementowy (3 przesunięcia, 3 obroty na osiach). 
To jest nasze $\delta x$!
  
Operator ^ (daszek) zamienia ten wektor na specjalną macierz antysymetryczną.

Z analizy wiemy, że dla bardzo małych wartości $x$, funkcja $e^x \approx 1 + x$ (rozwinięcie Taylora). 
W świecie macierzy wygląda to tak:
$$
\exp(\delta \boldsymbol{\xi}^\wedge) \approx \mathbf{I} + \delta \boldsymbol{\xi}^\wedge 
$$
I to jest klucz do wszystkich poniższych przekształceń.


**2. Target Pose Jacobian**

Let's find the Jacobian with respect to the target pose $\mathbf{T}_t$. 

We want to find how a small perturbation $\delta \boldsymbol{\xi}_t = [\mathbf{v}_t, \boldsymbol{\omega}_t]^T \in \mathbb{R}^6$ in the tangent space $\mathfrak{se}(3)$ affects the 3D point $P^t$. (We cant just calc derviatives with matrix like Se3 griup ):

$$
P^t = \mathbf{T}_t^{-1} \mathbf{T}_s P^s
$$

We perturb the target pose using right-multiplication:
$$ 
\mathbf{T}_{t, new} = 
\mathbf{T}_t \exp(\delta \boldsymbol{\xi}_t^\wedge) 
$$

Since the formula uses the inverse of $\mathbf{T}_t$, we apply the matrix inversion rule $(A \cdot B)^{-1} = B^{-1} \cdot A^{-1}$:
$$
\mathbf{T}_{t, new}^{-1} = 
(\mathbf{T}_t \exp(\delta \boldsymbol{\xi}_t^\wedge))^{-1} = 
\exp(\delta \boldsymbol{\xi}_t^\wedge)^{-1} \mathbf{T}_t^{-1} 
$$

For small perturbations, we use the first-order Taylor approximation for the Lie group inverse:
$$
\exp(\delta \boldsymbol{\xi}_t^\wedge)^{-1} \approx 
(\mathbf{I} - \delta \boldsymbol{\xi}_t^\wedge)
$$

Now, let's substitute this back into the equation for the perturbed point $P^t_{new}$:

$$
P^t_{new} \approx (\mathbf{I} - \delta \boldsymbol{\xi}_t^\wedge) \mathbf{T}_t^{-1} \mathbf{T}_s P^s
$$


Notice that $\mathbf{T}_t^{-1} \mathbf{T}_s P^s$ is exactly our original, unperturbed point $P^t$. 
Substituting this:
$$
P^t_{new} \approx (\mathbf{I} - \delta \boldsymbol{\xi}_t^\wedge) P^t = 
P^t - \delta \boldsymbol{\xi}_t^\wedge P^t
$$

The change in the point, $\Delta P^t = P^t_{new} - P^t$, is:
$$
\Delta P^t = - \delta \boldsymbol{\xi}_t^\wedge P^t
$$

To extract the Jacobian matrix, we need to separate the $6 \times 1$ vector $\delta \boldsymbol{\xi}_t$ from the operation. 

We decompose $\delta \boldsymbol{\xi}_t^\wedge P^t$ into translation $\mathbf{v}_t$ and rotation $\boldsymbol{\omega}_t$:

$$
\delta \boldsymbol{\xi}_t^\wedge P^t = 
\mathbf{v}_t + \boldsymbol{\omega}_t^\wedge P^t
$$

Using the cross-product property $\mathbf{a}^\wedge \mathbf{b} = -\mathbf{b}^\wedge \mathbf{a}$, we can rewrite the rotation part:
$$
\boldsymbol{\omega}_t^\wedge P^t = -(P^t)^\wedge \boldsymbol{\omega}_t
$$

Putting it all together into a matrix form multiplied by the perturbation vector $\delta \boldsymbol{\xi}_t$:
$$
\Delta P^t = 
- (\mathbf{v}_t - (P^t)^\wedge \boldsymbol{\omega}_t) = 
-\mathbf{I}_{3\times3}\mathbf{v}_t + (P^t)^\wedge \boldsymbol{\omega}_t
$$


$$
\Delta P^t = 
[-\mathbf{I}_{3\times3} \ | \ (P^t)^\wedge] \begin{bmatrix} \mathbf{v}_t \\ \boldsymbol{\omega}_t \end{bmatrix}
$$

Thus, the Jacobian matrix of $P^t$ with respect to the target pose perturbation $\delta \boldsymbol{\xi}_t$ is:

$$
\frac{\partial P^t}{\partial \delta \boldsymbol{\xi}_t} = 
[-\mathbf{I}_{3\times3} \ | \ (P^t)^\wedge]
$$

As | is connetion of to matrixes side by side, and ^ is skew-symmetric matrix,:

$$
\frac{\partial P^t}{\partial \delta \boldsymbol{\xi}_t} = 
\left[ \ \begin{array}{ccc|ccc} 
-1 & 0 & 0 & 0 & -z & y \\
0 & -1 & 0 & z & 0 & -x \\
0 & 0 & -1 & -y & x & 0
\end{array} \ \right]
$$

**2. Source Pose Jacobian**

Let's find the Jacobian with respect to the source pose $\mathbf{T}_s$. 

We want to find how a small perturbation $\delta \boldsymbol{\xi}_s = [\mathbf{v}_s, \boldsymbol{\omega}_s]^T \in \mathbb{R}^6$ in the tangent space $\mathfrak{se}(3)$ affects the 3D point $P^t$.

$$
P^t = 
\mathbf{T}_t^{-1} \mathbf{T}_s P^s
$$

We perturb the source pose using right-multiplication: 
$$ 
\mathbf{T}_{s, new} = \mathbf{T}_s \exp(\delta \boldsymbol{\xi}_s^\wedge) 
$$

Notice there is no inverse on $\mathbf{T}_s$ in our base equation. 
For small perturbations, we directly use the first-order Taylor approximation:
$$
\exp(\delta \boldsymbol{\xi}_s^\wedge) \approx (\mathbf{I} +
\delta \boldsymbol{\xi}_s^\wedge)
$$

Now, substitute this back into the equation for the perturbed point $P^t_{new}$:

$$
P^t_{new} \approx \mathbf{T}_t^{-1} \mathbf{T}_s (\mathbf{I} + \delta \boldsymbol{\xi}_s^\wedge) P^s
$$

Now, Distribute the multiplication over the parentheses:
$$
P^t_{new} \approx \mathbf{T}_t^{-1} \mathbf{T}_s P^s + 
\mathbf{T}_t^{-1} \mathbf{T}_s (\delta \boldsymbol{\xi}_s^\wedge P^s)
$$

The first term is exactly our original point $P^t$. 
Let's denote the relative transformation from source to target frame as $\mathbf{T}_{ts} = \mathbf{T}_t^{-1} \mathbf{T}_s$. 

The equation becomes:
$$
P^t_{new} \approx P^t + \mathbf{T}_{ts} (\delta \boldsymbol{\xi}_s^\wedge P^s)
$$

The change in the point is $\Delta P^t = P^t_{new} - P^t$:
$$
\Delta P^t = 
\mathbf{T}_{ts} (\delta \boldsymbol{\xi}_s^\wedge P^s) =
\mathbf{T}_t^{-1} \mathbf{T}_s (\delta \boldsymbol{\xi}_s^\wedge P^s)
$$

**Math trics**
Important Geometry Property: 
The term $(\delta \boldsymbol{\xi}_s^\wedge P^s)$ represents a small 3D displacement vector, not a 3D point. 

When a geometric transformation matrix ($\mathbf{T}_{ts}$) is applied to a displacement vector, the translational part of the matrix has no effect. 

Only the rotational part, $\mathbf{R}_{ts}$, rotates the vector. Therefore:
$$
\mathbf{T}_{ts} (\delta \boldsymbol{\xi}_s^\wedge P^s) = \mathbf{R}_{ts} (\delta \boldsymbol{\xi}_s^\wedge P^s)
$$

Now, just like before, we decompose the perturbation:
$$
\delta \boldsymbol{\xi}_s^\wedge P^s = \mathbf{v}_s + \boldsymbol{\omega}_s^\wedge P^s
$$

Using the cross-product property $\mathbf{a}^\wedge \mathbf{b} = -\mathbf{b}^\wedge \mathbf{a}$, we rewrite the rotation part:
$$
\boldsymbol{\omega}_s^\wedge P^s = -(P^s)^\wedge \boldsymbol{\omega}_s
$$
Putting it into matrix form separated from the perturbation vector $\delta \boldsymbol{\xi}_s$:

$$
\Delta P^t = \mathbf{R}_{ts} (\mathbf{I}_{3\times3} \mathbf{v}_s - (P^s)^\wedge \boldsymbol{\omega}_s) = \mathbf{R}_{ts} [\mathbf{I}_{3\times3} \ | \ -(P^s)^\wedge] \begin{bmatrix} \mathbf{v}_s \\ \boldsymbol{\omega}_s \end{bmatrix}
$$

Thus, the Jacobian matrix of $P^t$ with respect to the source pose perturbation $\delta \boldsymbol{\xi}_s$ is:

$$
\frac{\partial P^t}{\partial \delta \boldsymbol{\xi}_s} = 
\mathbf{R}_{ts} [\mathbf{I}_{3\times3} \ | \ -(P^s)^\wedge] = 
\mathbf{R}_{t}^{-1}\mathbf{R}_{s} [\mathbf{I}_{3\times3} \ | \ -(P^s)^\wedge] 
$$

Where:
$$
[\mathbf{I}_{3\times3} \ | \ -(P^s)^\wedge] = 
\left[ \ \begin{array}{ccc|ccc} 
1 & 0 & 0 & 0 & z^s & -y^s \\
0 & 1 & 0 & -z^s & 0 & x^s \\
0 & 0 & 1 & y^s & -x^s & 0
\end{array} \ \right]
$$
**Note:** This matrix HAVE to be rotated with $\mathbf{R}_{t}^{-1}\mathbf{R}_{s}$

**Elvation angle Jacobian **
Let's find the Jacobian with respect to the scalar elevation angle $\phi^s$.
This angle only affects the initial position of the point in the source frame, $P^s$. 
We want to find how a small change in $\phi^s$ affects the 3D point in the target frame, $P^t$.


Using the chain rule, we can split this into two parts:
$$
\frac{\partial P^t}{\partial \phi^s} = 
\frac{\partial P^t}{\partial P^s} \cdot \frac{\partial P^s}{\partial \phi^s}
$$

First expression:
$$
\frac{\partial P^t}{\partial P^s} =
\frac{\partial (\mathbf{R}_{ts} P^s + \mathbf{t}_{ts})}{\partial P^s} = 
\mathbf{R}_{ts} = 
\mathbf{R}_{t}^{-1}\mathbf{R}_{s}
$$

Second expression:
We take the definition of the inverse projection from Section 1 (NED frame):
$$P^s = \begin{bmatrix} x^s \\ y^s \\ z^s \end{bmatrix} = 
\begin{bmatrix} 
r^s \cos\phi^s \cos\theta^s \\ 
r^s \cos\phi^s \sin\theta^s \\ 
r^s \sin\phi^s 
\end{bmatrix}
$$

Now, we calculate the partial derivatives of each coordinate with respect to $\phi^s$:

$$
\frac{\partial x^s}{\partial \phi^s} = 
r^s (-\sin\phi^s) \cos\theta^s = 
-r^s \sin\phi^s \cos\theta^s
$$
$$
\frac{\partial y^s}{\partial \phi^s} = 
r^s (-\sin\phi^s) \sin\theta^s = 
-r^s \sin\phi^s \sin\theta^s
$$
$$
\frac{\partial z^s}{\partial \phi^s} = 
r^s \cos\phi^s
$$

Putting this into a column vector:
$$
\frac{\partial P^s}{\partial \phi^s} = 
\begin{bmatrix} 
-r^s \sin\phi^s \cos\theta^s \\ 
-r^s \sin\phi^s \sin\theta^s \\ 
r^s \cos\phi^s 
\end{bmatrix}
$$

Compising these two expression we get:
$$
\frac{\partial P^t}{\partial \phi^s} = 
\mathbf{R}_{ts} 
\begin{bmatrix} 
-r^s \sin\phi^s \cos\theta^s \\ 
-r^s \sin\phi^s \sin\theta^s \\ 
r^s \cos\phi^s 
\end{bmatrix}
$$


5. Putting all together:

Target pose Jacobian: (Size $2 \times 6$):
$$ 
\mathbf{J}_{\mathbf{T}_t} = \frac{\partial e}{\partial \delta \boldsymbol{\xi}_t} = -\mathbf{J}_\Pi \ [-\mathbf{I}_{3\times3} \ | \ (P^t)^\wedge] 
$$
Source pose Jacobin (size $2 \times 6$):
$$ 
\mathbf{J}_{\mathbf{T}_s} = \frac{\partial e}{\partial \delta \boldsymbol{\xi}_s} = -\mathbf{J}_\Pi \ \mathbf{R}_{ts} [\mathbf{I}_{3\times3} \ | \ -(P^s)^\wedge] 
$$
Elevation angle jacobian ( Size) $2 \times 1$):
$$ 
\mathbf{J}_{\phi^s} = \frac{\partial e}{\partial \phi^s} = -\mathbf{J}_\Pi \ \mathbf{R}_{ts} \begin{bmatrix} -r^s \sin\phi^s \cos\theta^s \\ -r^s \sin\phi^s \sin\theta^s \\ r^s \cos\phi^s \end{bmatrix} 
$$





