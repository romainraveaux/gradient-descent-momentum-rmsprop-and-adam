# Gradient Descent : Momentum, RMSProp and ADAM

## Author : Romain Raveaux
Gradient descent is an algorithm to find the parameters of a model.

On the basis of a set of data set comprising M input values $X = (x_1, . . . , x_M)^T$ and their corresponding target values $T = (t_1, . . . , t_M)^T$, we want to find the parameters $W$ of the model $f(x,W)$ such that a given criterion is minimized. 

## The criterion

In machine learning, the criterion is often an error function such that : 
$$L(X,T,W)=\frac{1}{M}\sum_{n=1}^M { \{ f(x_n,W)-t_n }\}^2 $$
$$or$$
$$L(X,T,W)=\frac{-1}{M}\sum_{n=1}^M \{   t_n \log (f(x_n,W))  +  (1-t_n) \log( 1-f(x_n,W) )  \} \quad \text{with }t_n \in \{0,1\}$$

## The optimization problem

The optimization problem is then defined as :
$$ W^*=arg \min_W L(X,T,W)$$

We are seeking for the minimum of the function $L$ so we need to find where the derivative of the function is equal to zero:
$$\dfrac{\partial L(X,T,W)}{\partial W} = 0 $$ 
This can be done by the gradient descent method. 


## Gradient descent method
If the fonction $L$ is convex then we can start with any random parameters $W$ and reach the global minimum of $L$ by gradient descent. If the the problem is not convex then the gradient descent method only finds a suboptimal solution.

The goal is to choose the parameter update to comprise a small step in the direction of the negative gradient, so that
$$W^{(t+1)}=W^{(t)}-\alpha. \dfrac{\partial L}{\partial W^{(t)}}$$
where the parameter $\alpha > 0$ is known as the learning rate.


## Issue of the Gradient descent

The gradient on its own can be a bit noisy it means it changes a lot at each iteration. In order to get a clear trend on the gradient values, it is possible to make it smoother.
