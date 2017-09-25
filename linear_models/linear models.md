# Linear Models

## Linear Regression

### Model

In linear regression, we decide to approximate `$y$` as a linear function of `$x$`:

```math
h_{\theta}=\theta_0+\theta_1x_1+\theta_2x_2 +...+\theta_nx_n=\sum_{i=0}^n\theta_ix_i=\theta^Tx,

\theta_i:\ patameters\ also\ called\ weights

x_0=1
```
where `$\theta$` and `$x$` are both vectors and `$n$` is the number of input variables(features)

### Strategy

Given a training set, one reasonable method of picking or learning the parameters `$\theta$` seems to make `$h(x)$` close to y, at least for the training examples in training set we have. So as a natural choice, we define the function that measures, for each of the `$\theta's$`, how close the `$h(x^{(i)})'s$` are to the corresponding `${y^{(i)}}'s$` called **cost function**:

```math
J(\theta)\ = \ \tfrac{1}{2}\sum_{i=1}^m(h_\theta(x^{(i)})\ - \ y^{(i)})^2
```

### Method

#### 1  LMS algorithm

Corresponding strategy, our goal is to choose reasonable `$\theta$` to minimize the cost function `$J(\theta)$`, namely **least mean square error**.

##### 1.1 Gradient descent

In gradient descent algorithm, we'll starts with some initial parameters `$\theta's$`, and repeatedly performs the update:

```math
\theta_j\ :=\ \theta_j\ -\ \alpha\tfrac{\partial}{\partial\theta_j}J(\theta)
```

Here, `$\alpha$` is called the **learning rate**. This is a very natural algorithm that repeatedly take a step in the direction of deepest decrease of J, in other words, gradient direction. In order to implement this algorithm, lets first work it out for the case that if we have only one traning example `$(x,\ y)$`, so that we can neglect the sum in the definition of `$J$`. Consider the partial derivative term:

```math
\tfrac{\partial}{\partial\theta_j}J(\theta)\ = \ \tfrac{\partial}{\partial\theta_j}\tfrac{1}{2}(h_\theta(x)\ - \ y)^2

=\ 2\ \times\ \tfrac{1}{2}(h_\theta(x)\ - \ y)\ \cdot\ \tfrac{\partial}{\partial\theta_j}(\theta^Tx\ - \ y)

=\ (h_\theta(x)\ -\ y)x_j
```

So for a single training example, e.g. *i-th* this gives the update rule:

```math
\theta_j\ :=\ \theta_j\ -\ (h_\theta(x)\ -\ y)x_j
```

The rule is called the LMS update rule which stands for "least mean squares", and also known as the **Widrow-Hoff** learning rule.

Now we'd derived the **LMS** rule based on the case where there was only a single training example. Generally, there are two way to modify this method for a training set of more than one example.

##### BGD(batch gradient descent)

The first method is replace the case above with the following algorithm:

```math
Repeat\ until\ convergence \{

    \theta_j\ :=\ \theta_j - \alpha\sum^m_{i=1}(h_\theta(x^{(i)})\ -\ y^{(i)})x\ (for\ every\ j)

\}
```

We can verify that the quantity in the summation in the update rule is just `$\tfrac{\partial}{\partial\theta_j}J(\theta)$` for the original definition of `$J$`. This method lools at every example in the entire training set on every step, so it is called **batch gradient descent**. Note that, while gradient descent can be susceptible to local mimima in general, the optimization problem we have posed here for linear regression has only one global, and no other local, optima; thus gradient descent always converges to the global minimum because the cost function `$J$` is a convex quadratic function. As we all know, all optimization problems of convex function have only a global optima.

##### SGD(stochastic gradient descent)

There is an alternative to batch gradient descent that also works well.

```math
Loop\ \{

    for\ i\ =\ 1\ to\ m, 
    
    \{
    
        \theta_j\ :=\ \theta_j - \alpha(h_\theta(x^{(i)})\ -\ y^{(i)})x^{i}_j\ (for\ every\ j) 
    
    
    \}
    
\}
```

In this algorithm, we repeatedly run through thet training set, and each time we encounter a training example, we update the parameters according to the gradient o f the error with respect to that single training example only. So this algrithm is called **stochastic gradient descent**. Often, stochatic gradient descent gets `$\theta$` "close" to the minimum much faster than batch gradient descent. Particularly when the training set is large, stochastic gradient descent is often preferred over batch gradient descent.

##### 1.2 The normal equations

consider:

```math
\tfrac{1}{2}(X\theta\ -\ y)^T(X\theta\ -\ y)\ =\ \tfrac{1}{2}\sum_{i=1}^{m}(h_\theta(x^{(i)})\ -\ y^{(i)})^2\ =\ J(\theta)
```

hence:

```math
\nabla_{\theta}J(\theta)\ =\ \tfrac{1}{2}\nabla_{\theta}(\theta^TX^TX\theta\ -\ \theta^TX^Ty\ -\ y^TX\theta\ +\ y^Ty)

=\ \tfrac{1}{2}\nabla_{\theta}tr(\theta^TX^TX\theta\ -\ \theta^TX^Ty\ -\ y^TX\theta\ +\ y^Ty)

=\ \tfrac{1}{2}(X^TX\theta\ +\ X^TX\theta\ - 2X^Ty)

=\ X^TX\theta\ -\ X^Ty
```

To minimize `$J$`, we set its derivations to zero, and obtain the **normal equation**:

```math
    X^X\theta\ =\ X^Ty
```

Thus, the value of `$\theta$` that minimizes `$J(\theta)$` is given in closed form by the equation

```math

\theta\ =\ (X^TX)^{-1}X^Ty
```

### Probabilistic interpretation

Here we will give a set of probabilistic assumptions, under which least-squares regression is derived as a very natural algorithm.

Let's assump that the target and inputs are related via the equation:

```math
y^{(i)}\ =\ \theta^Tx^{(i)}\ +\ \epsilon^{(i)}
```

where `$\epsilon{(i)}$` is an error term that captures either unmodeled effects such as there are some very pertinent to predicting the target variable but we'd left out of the regression, or random noise. Let's further assume that the `$\epsilon{(i)}$` is distributed **IID** according to a **Gaussian** distribution also called a **Normal** distribution with mean zero and some variance `$\sigma^2$`. So We can write this assumption as `$\epsilon{(i)}\ \sim\ N(0, \sigma^2)$`. I.e., the density of `$\epsilon{(i)}$` is given by:

```math
p(\epsilon^{(i)})\ =\ \tfrac{1}{\sqrt{2\pi}}e^{-\tfrac{(\epsilon^{(i)})^2}{2\sigma^2}}
```

This implies that

```math
p(y^{(i)}|x^{(i)};\theta)\ =\ \tfrac{1}{\sqrt{2\pi}}e^{-\tfrac{(y^{(i)}\ -\ \theta^Tx^{(i)})^2}{2\sigma^2}}
```

Given `$X$` and parameter vector `$\theta$`, the probability of the data is given by `$p(y|X;\theta)$`, and we instead call if the **likehood** function:

```math
L(\theta)\ =\ L(\theta;X,y)\ = \ \prod_{i=1}^m\tfrac{1}{\sqrt{2\pi}}e^{-\tfrac{(y^{(i)}\ -\ \theta^Tx^{(i)})^2}{2\sigma^2}}
```

We will use the principal of **maximum likehood** to choose the best parameter `$\theta$` which says that we should choose `$\theta$` so as to make the data as high probability as possible. I.e., we should choose `$\theta$` to maximize `$L(\theta)$`.

Instead of maximizing `$L(\theta)$`, we can also maximize any strictly increasing function of `$L(\theta)$`. In particular, the derivations will be a bit simpler if we instead maxmize the **log likehood** `$l(\theta)$`:

```math
l(\theta)\ =\ logL(\theta)\ =\ log \prod_{i=1}^m\tfrac{1}{\sqrt{2\pi}}e^{-\tfrac{(y^{(i)}\ -\ \theta^Tx^{(i)})^2}{2\sigma^2}}

=\ \sum_{i=1}^mlog\tfrac{1}{\sqrt{2\pi}}e^{-\tfrac{(y^{(i)}\ -\ \theta^Tx^{(i)})^2}{2\sigma^2}}

=\ mlog\tfrac{1}{\sqrt{2\pi}}\ - \tfrac{1}{\sigma^2}(\tfrac{1}{2}\sum_{i=1}^m(y^{(i)}\ -\ \theta^Tx^{(i)})^2)
```

Hence, maximizing `$l(\theta)$` gives the same answer as minimizing 

```math
\tfrac{1}{2}\sum_{i=1}^m(y^{(i)}\ -\ \theta^Tx^{(i)})^2
```

which we recognize to be `$J(\theta)$`, out original least-squares cost function.

## Classificaition and logistic regression

First, we will focus on the **binary classification** problem in which y can be take on only two values, 0 and 1. 0 is also called ***negative class*, and 1 the **positive class**.

### Model

We could approach the classification problem ignoring the fact that `$y$` is discrete-valued, and use our old linear regression algorithm to try to predict `$y$` given x.

To fit the fact that `$y\ \in\ \{0,\ 1\}$`, let's change the form of our hypotheses `$h_\theta(x)$` which can map `$\theta^Tx$` from space `$R$` to interval `$[0,\ 1]$`, we will choose

```math
h_\theta(x)\ =\ g(\theta^Tx)\ =\ \tfrac{1}{1\ +\ e^{-\theta^Tx}}
```

where

```math
g(z)\ =\ \tfrac{1}{1\ +\ e^{-z}}
```

is called the **logistic function** or the **sigmoid function** which tends towards 1 as `$z\ \to\ +\infty$` and tends towards 0 as `$z\ \to\ -\infty$`. Before moving on, here's a useful property of the derivative of the sigmoid function:

```math
g'\ =\ g'(z)\ =\ \tfrac{1}{(1\ +\ e^{-z})^2}e^{-z}

=\ \tfrac{1}{(1\ +\ e^{-z})^2}(1\ -\ \tfrac{1}{(1\ +\ e^{-z})^2})

=\ g(z)(1\ -\ g(z))
```

### Strategy

We know that least squares regression could be derived as the maximum likehood estimator under a set of assumptions, so in logistic regression, let's endow our classification model with a set of probabilistic assumptions, and then fit the parameters via maximum likehood estimator.

Here we assume that `$y|x;\theta$` is distributed as **Bernoulli** distribution

```math
p(y^{(i)}=1|x^{(i)};\theta)=h_\theta(x^{(i)})

p(y^{(i)}=0|x^{(i)};\theta)=1-h_\theta(x^{(i)})

```

Note that this can be written more compactly as 

```math
p(y^{(i)}|x^{(i)};\theta)=(h_\theta(x^{(i)}))^{y^{(i)}}(1-h_\theta(x^{(i)}))^{(1-y^{(i)})}
```

Assuming that the m training examples were generated indepently, we can then write down the likehood of the parameters as

```math
L(\theta)\ =\ \prod_{i=1}^{m}p(y^{(i)}|x^{(i)};\theta)\ =\ \prod_{i=1}^{m}(h_\theta(x^{(i)}))^{y^{(i)}}(1-h_\theta(x^{(i)}))^{(1-y^{(i)})}
```

As before, it's easier to maximize the log likehood:

```math
l(\theta)\ =\ logL(\theta)\ =\ \sum_{i=1}^{m}y^{(i)}logh_\theta(x^{(i)})\ + \ (1\ -\ y^{(i)})log(1\ -\ h_\theta(x^{(i)}))
```

Then what we need to do is choose `$\theta$` to maximize the log likehood function. I.e., we need to minimize the **minus log likehood** `$-l(\theta)$`, and we can consider it as a cost function of logistic regression.

### Method

#### Gradient ascent

To get the parameters, we will maximize the log likehood `$l(\theta)$` or minimize the minus log likehood function. Similar to our derivation in the case of linear regression, we can use gradient ascent(corresponding to maximum likehood estimator). Written in vectorial notation, our updates will therefore be given by `$\theta:=\theta+\alpha\nabla{l(\theta)}$`.(Note the positive rather than negative sing in the update formula, since we're maximizing, rather than minimizing, a function now.) Let's start by working with just one training example `$(x, y)$`, and take derivatives to derive the stochastic gradient ascent rule:

```math
\tfrac{\partial}{\partial\theta_j}l(\theta)\ =\ (y\tfrac{1}{g(\theta^Tx)}\ -\ (1\ -\ y)\tfrac{1}{1\ -\ g(\theta^Tx)})g'(\theta^Tx)

=\  (y\tfrac{1}{g(\theta^Tx)}\ -\ (1\ -\ y)\tfrac{1}{1\ -\ g(\theta^Tx)})g(\theta^Tx)(1\ -\ g(\theta^Tx))x_j

=\ y(1\ -\ g(\theta^Tx))\ -\ (1\ -\ y)g(\theta^Tx)x_j

=\ (y\ -\ g(\theta^Tx))x_j\ =\ (y\ -\ h_\theta(x))x_j
```

This therefore gives us the stochastic gradient ascent rule:

```math
\theta_j:=\theta_j\ +\ \alpha(y^{(i)}\ -\ h_\theta(x^{(i)}))x^{(i)}_j
```

#### Newton

First, Let's consider Newton's method for finding a zero of a function. Specifically, suppose we have some function `$f\ :\ R \to R$`, and we wish to find a value of `$\theta$` so that `$f(\theta)\ =\ 0$`. Here, `$\theta\ \in\ R$` is a real number. Newton's method performs the following update:

```math
\theta:=\theta-\tfrac{f(\theta)}{f'(\theta)}
```

This method has a natural interpretation in which we can think of it as approximating the funtion `$f$` via a linear funtion that is tangent to `$f$` at the current guess `$\theta$`, solving for where linear function equals to zero, and letting the next guess for `$\theta$` be that linear function is zero. Here, we can write the tangent line:

```math
f(\theta)\ =\ f'(\theta_0)\theta\ +\ f(\theta_0)

```

where `$\theta_0$` is current guess of `$\theta$`

Then, let the tangent line equation be zero, we can get:

```math
\theta=\theta_0-\tfrac{f(\theta_0)}{f'(\theta_0)}
```

So Newton method gives a way to getting to `$f(\theta)=0$`. The maxima of `$l$` correspond to points where its first derivative `$l'(\theta)$` is zero. So by letting `$f(\theta)=l'(\theta)$`, we can use the same algorithm to maximize `$l$`, and obtain the update rule:

```math
\theta\ :=\ \theta\ -\ \tfrac{l'(\theta)}{l''(\theta)}
```
Lastly, in our problems, `$\theta$` is a vector-valued, so we need to generalize Newton's method to this setting. The generalization of Newton's method to this multidimentional setting also called the **Newton-Raphson** method is given by:

```math
\theta\ :=\ \theta\ -\ H^{-1}\nabla_\theta{l(\theta)}
```

where, `$\nabla_\theta{l(\theta)}$`, as usual, the vector of partial derivative of `$l(\theta)$`; and `$H$` is an `$n-by-n$` matrix called *Hessian*, whose entries are given by:

```math
H_{ij}\ =\ \tfrac{\partial^2l(\theta)}{\partial\theta_i\partial\theta_j}
```
Newton’s method typically enjoys faster convergence than (batch) gradient descent, and requires many fewer iterations to get veryclose to the minimum. One iteration of Newton’s can, however, be more expensive than one iteration of gradient descent, since it requires finding and inverting an n-by-n Hessian; but so long as n is not too large, it is usually much faster overall. When Newton’s method is applied to maximize the logistic regression log likelihood functionℓ(θ), the resulting method is also called **Fisher scoring**.

## Generalized Linear Model

So far, we have seen a regression example, a classification example. In the regression example, we had `$y|x;\theta\sim{N(\mu,\sigma^2)}$`, and in the classification one, `$y|x;\theta\sim{Bernouli(\phi)}$`, where for some appropriate definitions of `$\mu$` and `$\phi$` as functions of `$x$` and `$\theta$`. In fact, both of these methods are special cases of a broader family of models, called **Generalized Linear Models(GLMs)**

### The exponential family

To work our way up to GLMs, we will begin defining exponential family distributions which can be written in the form

```math
p(x;\eta)\ =\ b(y)e^{\eta^TT(y)\ -\ a(\eta)}
```
Here, `$\eta$` is called the **natrual parameter** (also called the **canonical parameter**) of the distribution; `$T(y)$` is the **sufficient statistic** (often `$T(y)=y$`); and `$a(\eta)$` is the **log partition function**. The quantity `$e^{-a(\eta)}$` essentially plays the role of a normalization constant, that makes sure the distribution `$p(y, \eta)$` sums/integrates over y to 1.

A fixed choice of `$T$`, `$a$` and `$b$` defines a family (or set) of distributions that is parameterized by `$\eta$`; as we vary `$y$`, we then get different distribution within this family.

Now we show the the Bernouli and Gaussian distributions are example of exponential family distributions.

#### Bernouli example

We can write the Bernouli distribution as:

```math
p(y; \phi)\ =\ \phi^y(1-\phi)^{1-y}

=e^{ylog\phi\ +\ (1-y)log(1-\phi)}

=e^{log\tfrac{\phi}{1-\phi}y\ +\ log(1-y)}
```

Thus, the natrual parameter is given by `$\eta\ =\ log(\phi/(1-\phi))$`. To complete the formulation of the Bernouli distribution as an exponential family distribution, we also have

```math
\phi\ =\ \tfrac{1}{1+e^{-\eta}}

T(y)\ =\ y

a(\eta)\ =\ -log(1-\phi)
=log(1+e^\eta)

b(y)\ =\ 1
```
#### Gaussian example

As the same as deriving linear regression, let set `$\sigma^2=1$`. So we have

```math
p(y;\eta)\ =\ \tfrac{1}{\sqrt{2\pi}}e^{-\tfrac{1}{2}(y-\mu)^2}

=\ \tfrac{1}{\sqrt{2\pi}}e^{-\tfrac{1}{2}y^2}e^{\mu{y}-\tfrac{1}{2}\mu^2}
```

Thus, we see that Gaussian is in the exponential family, with

```math
\eta\ =\ \mu

T(y)\ =\ y

a(\eta)\ =\ \tfrac{\mu^2}{2}\ =\ \tfrac{\eta^2}{2}

b(y)\ =\ \tfrac{1}{\sqrt{2\pi}}e^{-\tfrac{y^2}{2}}
```

There're many other distributions that are members of the exponential family: The multinomial (which we’ll see later), the Poisson (for modelling count-data; also see the problem set); the gamma and the exponential (for modelling continuous, non-negative random variables, such as time-intervals); the beta and the Dirichlet (for distributions over probabilities); and many more.

### GLMs

More generally, consider a classification or regression problem where we would like to predict the value of some random variable `$y$` as a function of `$x$`. To derive a GLM for this problem, we will make the following three assumptions about the conditional distribution of `$y$` given `$x$` and about our model:

* **1**. `$y|x;\theta\sim{ExponentialFamily(\eta)}$`. I.e., given `$x$` and `$\theta$`, the distribution of `$y$` follows exponential family distrubition, with parameter `$\eta$`
* **2**. Given x, out goal is to predict the expected value of `$T(y)$`. In most of our examples, we will have `$T(y)=y$`, so this means we would like the prediction `$h(x)$` output by out learned hypothesis `$h$` to satisfy `$h(x)=E[y|x]$`.
*  **3**. The natural parameter `$\eta$` and the inputs x are related linearly: `$\eta=\theta^Tx$`. (Or, if `$\eta$` is vector-valued, then `$\eta_i=\theta_i^Tx$`)

These three assumptions/design choices will allow us to derive a very elegant class of learning algorithms, namely GLMs, that have many desirable properties such as ease of learning. Furthermore, the resulting models are often very effective for modelling different types of distributions over `$y$`.

#### Example1: Ordinary Least Squares

To show that ordinary least squares is a special case of the LGM family of model, consider the setting where the target variable `$y$` (also called the **response variable** in GLM terminology) is continuous, and we model the conditional distribution of `$y$` given `$x$` as a Gaussian `$N(\mu,\sigma^2)$`. As we saw previously, in the formulation of the Gaussian as an exponential family distribution, we had `$\mu=\eta$`. So, we have

```math
h_\theta(x)\ =\ E[y|x;\theta]

=\mu

=\eta

=\theta^Tx
```

The first equality follows from Assumption 2 above; the second equality follows from the fact that `$y|x;\theta\sim{N(u,\sigma^2)}$`, and so its expected value is given by `$\mu$`; the third equality follows from Assumption 1 (and our earlier derivation showing that `$\mu=\eta$` in the formulation of the Gaussian as an exponential family distribution); and the last equality follows from Assumption 3.

#### Examples2: Logistic Regression

In our formulation of the Bernoulli distribution as an exponential family distribution, we had `$\phi=\tfrac{1}{1+e^{-\eta}}$`. Furthermore, note that if `$y|x;\theta\sim{Bernoulli(\phi)}$`, then `$E[y|x;\theta]=\phi$`. So, following a similar derivation as the one for ordinary least squares, we get:

```math
h_\theta(x)\ =\ E[y|x;\theta]

=\phi

=\tfrac{1}{1+e^{-\eta}}

=\tfrac{1}{1+e^{-\theta^Tx}}
```

And the function `$g$` giving the distribution’s mean as a function of the natural parameter `$(g(\eta)=E[T(y);\eta])$` is called the **canonical response function**. Its inverse, `$g^{-1}$` , is called the **canonical link function**.  Thus, the canonical response function for the Gaussian family is just the identify function; and the canonical response function for the Bernoulli is the logistic function.

## Softmax(Extension of GLMs)

Let's consider a classification problem in which the reponse variable `$y$` can take on any one of `$k$` values, so `$y\in\{1, 2, ..., k\}$`. The response variable is still discrete, but can now take on more than two values. So we will thus model it as distributed according to a multinomial distribution.

To do so, we'll begin by expressing the multinomial distribution as an exponential family distribution. To parameterize a multinomial over `$k$`, one could use `$k$` parameters `$\phi_1, \phi_2,...,\phi_k$` specifying the probability of the outcomes. However, these parameters is rebudant, they would be independent(since `$\sum_{i=1}^k\phi_i=1$`). So we will instead parameter the multinomial distribution with only `$k-1$` parameters `$\phi_1, \phi_2, ..., \phi_{k-1}$`, where `$\phi_i=p(y=i;\phi)$` and `$p(y=k;\phi)=1-\sum_{i=1}^{k-1}$`. For notational convenience, we will also let `$\phi_k=1-\sum_{i=1}^{k-1}$`, but this `$\phi_k$` is not a parameter.

To express the multinomial as an exponential distribution, we will define `$T(y)\in{R^{k-1}}$` as follows:

```math
T(1)=[1, 0, 0, ..., 0]^T

T(2)=[0, 1, 0, ..., 0]^T

T(3)=[0, 0, 1, ..., 0]^T

...

T(k-1)=[0, 0, 0, ..., 1]^T

T(k)=[0, 0, 0, ..., 0]^T
```

We introduce one more very useful piece of notation. An indicator function `$1\{\bullet\}$` takes on a value of 1 if argument is true and 0 otherwise (`$1\{True\}=1, 1\{False\}=0$`). So we can also write the relationship between `$T(y)$` and `$y$` as `$T(y)_i=1\{y=i\}$`. Further, we have that `$E[T(y)_i]=1\{y=i\}=\phi_i$`

Now we can derive that multinomial distribution is a member of the exponential family:

```math
p(y;\phi)\ =\ \phi_1^{1\{y=1\}}\phi_2^{1\{y=2\}}...\phi_k^{1\{y=k\}}

=\ \phi_1^{1\{y=1\}}\phi_2^{1\{y=2\}}...\phi_k^{1-\sum_{i=1}^{k-1}1\{y=i\}}

=\ \phi_1^{T(y)_1}\phi_2^{T(y)_2}...\phi_k^{1-\sum_{i=1}^{k-1}T(y)_i}

=\ e^{T(y)_1log(\phi_1)\ +\ T(y)_2log(\phi_2)\ +\ ...\ + \  (1-\sum_{i=1}^{k-1}T(y)_i)log(\phi_k)}

= e^{T(y)_1log(\phi_1/\phi_k)\ +\ T(y)_2log(\phi_2/\phi_k)\ +\ ...\ +\  log(\phi_k)}

=b(y)e^{\eta^TT(y)\ -\ a(\eta)}
```

where

```math
\eta=[log(\phi_1/\phi_k), log(\phi_2/\phi_k),...,log(\phi_{k-1}/\phi_k)]

a(\eta)=-log(\phi_k)

b(y)=1
```

The link function is given by:
```math
\eta_i\ =\ log(\phi_i/\phi_k)
```

For convinience, we have also defined `$\eta_k=log(\phi_k/\phi_k)=0$`. To invert the link function and derive the response function. We therefore have that

```math
\phi_i\ =\ \phi_ke^{\eta_i}

\phi_k\sum_{i=1}^ke^{\eta_i}\ =\ \sum_{i=1}^k\phi_i\ =\ 1

\phi_k\ =\ 1/\sum_{i=1}^ke^{\eta_i}
```

So we can get the response function:

```math
\phi_i\ =\tfrac{e^{\eta_i}}{\sum_{j=1}^ke^{\eta_j}}
```

The function mapping from the `$\eta$`'s to `$\phi$`'s is called **softmax** function.

### Model 

We use the assumption 3 that the `$\eta_i$`'s are linearly related to the `$x$`'s. So we have `$\eta_i=\theta_i^Tx$` (for `$i=1,2,...,k-1$`). And we also define `$\theta_k=0$`, so that `$\eta_k=0$`. So our model assumes that the conditional distribution of `$y$` given x is given by:

```math
p(y=i|x;\theta)\ =\ \phi_i

=\tfrac{e^{\eta_i}}{\sum_{i=1}^ke^{\eta_i}}

=\tfrac{e^{\theta_i^Tx}}{\sum_{j=1}^ke^{\theta_j^Tx}}
```

This model is called **softmax regression**, and the hypothesis will output:

```math
h_\theta(x)\ =\ E[T(y)|x;\theta]

=\ [\phi_1, \phi_2,...,\phi_{k-1}]^T

=\ [e^{\theta_1^Tx/\sum_{j=1}^ke^{\theta_j^Tx}}, e^{\theta_2^Tx/\sum_{j=1}^ke^{\theta_j^Tx}},...,e^{\theta_{k-1}^Tx/\sum_{j=1}^ke^{\theta_j^Tx}}]^T
```

### Strategy

Similar to our original derivation of ordinary least square and logistic regression, if we have a training set of `$m$` examples `$\{(x^{(i),y^{(i)}});i=1,2,...,m\}$` and would like to learn the parameters `$\theta_i$` of this model, we would writing down the log-likehood

```math
l(\theta)\ =\ \sum_{i=1}^mp(y^{(i)}|x^{(i)};\theta)

=\ \sum_{i=1}^mlog\prod_{l=1}^k(e^{\theta_l^Tx^{(i)}}/\sum_{j=1}^ke^{\theta_j^Tx^{(i)}})^{1\{y^{(i)}=l\}}
```

In general, we can choose minus log-likehood function as the loss function. Here, we write it's loss function:

```math
J(\theta)\ =\ -\sum_{i=1}^m\sum_{l=1}^ky_l^{(i)}log\phi_l
```

where `$\phi_l=e^{\theta_l^Tx}/\sum_{j=1}^ke^{\theta_j^Tx}$`, which is called **cross-entropy** loss.

### Method

First of all, let's compute softmax's gradient. For convinience, we note that 

```math
J(\theta)\ =\ \tfrac{1}{m}\sum_{i=1}^mJ^{(i)}(\theta)

J^{(i)}\ =\ -\sum_{l=1}^ky_l^{(i)}log\phi_l

\phi_l\ =\ e^{f_l}/\sum_{j=1}^ke^{f_j}

f_j\ =\ \theta_l^Tx^{(i)}
```

So according to chain rule, we can get 

```math
\partial{J^{(i)}}/\partial{\theta_s}\ =\ -\sum_{l=1}^k\tfrac{\partial{J^{(i)}}}{\partial{\phi_l}}\tfrac{\partial{\phi_l}}{\partial{f_s}}\tfrac{\partial{f_s}}{\partial{\theta_s}}
```

Obviously, 

```math
\tfrac{\partial{J^{(i)}}}{\partial{\phi_l}}\ =\ y_l^{(i)}\tfrac{1}{\phi_l}

\tfrac{\partial{f_s}}{\partial{\theta_s}}\ =\ \nabla_{{(\theta_s^T)}^T}\theta_l^Tx^{(i)}\ =\ \nabla_{{(\theta_s^T)}^T}tr\theta_s^Tx^{(i)}\ =\ {(\nabla_{\theta_s^T}\theta_s^Tx^{(i)})}^T\ =\ {({x^{(i)}}^T)}^T\ =\ x^{(i)}
```

So, the key is deriving `$\tfrac{\partial{\phi_l}}{\partial{f_s}}$`. And we need to discuss it according to two cases.

When `$l=s$`:

```math
\tfrac{\partial{\phi_l}}{\partial{f_s}}\ =\ \tfrac{\partial}{\partial{f_s}}{(e^{f_s}/\sum_{j=1}^ke^{f_j})}

=\ \tfrac{e^{f_s}\sum_{j=1}^ke^{f_j}-e^{f_s}e^{f_s}}{{(\sum_{j=1}^ke^{f_j})}^2}

=\ \phi_s\ -\ \phi_s^2
```

when `$l\ne{s}$`:

```math
\tfrac{\partial{\phi_l}}{\partial{f_s}}\ =\ \tfrac{\partial}{\partial{f_s}}{(e^{f_l}/\sum_{j=1}^ke^{f_j})}

=\ \tfrac{-e^{f_l}e^{f_s}}{{(\sum_{j=1}^ke^{f_j})}^2}

=\ -\phi_l\phi_s
```

So

```math
\partial{J^{(i)}}/\partial{\theta_s}\ =\ -\sum_{l=1}^k\tfrac{\partial{J^{(i)}}}{\partial{\phi_l}}\tfrac{\partial{\phi_l}}{\partial{f_s}}\tfrac{\partial{f_s}}{\partial{\theta_s}}

=\ {(-\sum_{l\ne{s}}y_l^{(i)}{(-\phi_l\phi_s)/\phi_l}\ +\ y_s^{(i)}{(\phi_s\ -\ \phi_s^2)/\phi_s})}x^{(i)}

=\ {(\sum_{l=1}^ky_l^{(i)}\phi_s\ -\ y_s^{(i)})}x^{(i)}

={({h_\theta(x^{(i)})}_s\ -\ y_s^{(i)})}x^{(i)}
```
Now we can use a method such as gradient descent or Newton's method to get the parameters. For example, `$\theta$` can be update with rule below (BGD):

```math
\theta_s\ =\ \theta_s\ -\ \alpha\tfrac{1}{m}\sum_{i=1}^m{({h_\theta(x^{(i)})}_s\ -\ y_s^{(i)})}x^{(i)}
```
And SGD can be written as:

```math
\theta_s\ =\ \theta_s\ -\ \alpha{({h_\theta(x^{(i)})}_s\ -\ y_s^{(i)})}x^{(i)}
```
