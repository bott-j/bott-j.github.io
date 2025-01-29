---
layout: post
title: Monte-Carlo Integration
tags: 
- Algorithms
- Mathematics
- Python
- Statistics
image: /assets/posts/monte-carlo-integration-in-python/monte-carlo-integration.png
---

As I plan to implement a Monte-Carlo localisation algorithm following recent work with the HLS-LFCD2 LiDAR module, I thought it would be useful to revise some Monte-Carlo applications. Monte-Carlo methods are numerical methods which use repeated random samples to solve problems which may otherwise be difficult to solve analytically. In this post I will provide some Python examples and mathematical derivations for the type of Monte-Carlo integration methods typically encountered in engineering and mathematics. 

**Contents**
* [Using a Monte-Carlo Method to Calculate $$ \pi $$](#using-a-monte-carlo-method-to-calculate-pi)
* [The Monte-Carlo Estimator](#the-monte-carlo-estimator)
* [Standard Error of the Monte-Carlo Estimator](#standard-error-of-the-monte-carlo-estimator)
* [Using the Monte-Carlo Estimator to Calculate Definite Integrals](#using-the-monte-carlo-estimator-to-calculate-definite-integrals)
* [Using the Monte-Carlo Estimator to Calculate Double Integrals](#using-the-monte-carlo-estimator-to-calculate-double-integrals)
* [Recursive Stratified Sampling](#recursive-stratified-sampling)
* [Python Notebook and GitHub Repository](#python-notebook-and-github-repository)
* [References](#references)

# Using a Monte-Carlo Method to Calculate $$ \pi $$

To start with, a problem often seen in introductory Monte-Carlo texts is the numerical calculation of $$ \pi $$. We can calculate the value of Pi from the number of randomly sampled points within a two-dimensional plane that fall inside a circle. If we sample $$ N $$ such points from a uniform distribution, within a square range enclosing a circle, we would expect the probability of a point falling within the circle to depend on the ratio of the area of the circle and the square. Likewise, the ratio of area might be found from the probability of a point being found within the circle, which is how we estimate the value of $$ \pi $$.

The ratio of area between a circle of radius $$ r $$ and a square with sides of length $$ d $$ is:

$$ \frac{A_{circle}}{A_{square}} = \frac{\pi r^2}{d^2}$$

And for a square with sides of length 2 enclosing a unit circle with radius 1: 

$$ \frac{A_{circle}}{A_{square}} = \frac{1^2 \pi}{2^2} = \frac{\pi}{4}$$

Solving for $$ \pi $$, we find an expression in terms of the ratio of these areas: 

$$ \pi = 4\frac{A_{circle}}{A_{square}} $$

The Python implementation below samples 1'000 points from a uniform distribution within a square. The points are each tested against the equation of a circle to determine which points fall within the circle.

{% highlight python %}
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

# Create 1000 samples drawn from a uniform distribution 
df = pd.DataFrame(columns = ("X","Y"), data = np.random.uniform((0,0),(1,1),(1000,2)))
df['Inside'] = df.apply(lambda row: ((row["X"] - 0.5)**2 + (row["Y"] - 0.5)**2) <= 0.5**2, axis=1)

piCalculated = (df['Inside'].sum() / df.shape[0]) / (0.5**2)

# Summarise results 
print(f"After {df.shape[0]} trials, {df['Inside'].sum()} samples fall within the circle.")
print(f"The estimated value of Pi is: {piCalculated}")

# Create a plot
plt.scatter(df[df["Inside"] == False]["X"], df[df["Inside"] == False]["Y"], color="b", marker=".")
plt.scatter(df[df["Inside"] == True]["X"], df[df["Inside"] == True]["Y"], color="r", marker=".")
plt.xlabel("X")
plt.ylabel("Y")
plt.title("Estimating Pi (1'000 Samples)")
plt.show()
{% endhighlight %}

![Using Monte Carlo method, 1'000 points are sampled. Points which are red are inside the circle, blue points fall outside the circle.](/assets/posts/monte-carlo-method-of-integration-1.png?raw=true){: width="50%"}

As we get 777 out of 1000 points falling within the circle, the calculated value of Pi is:

$$ \pi = 4\frac{A_{circle}}{A_{square}} = 4\frac{777}{1000} = 3.108 $$

Only the first two digits are correct, however as we increase the number of samples, by the law of large numbers we could expect to get an estimate closer to the true value of $$ \pi $$.

The following example uses 100'000 sample points.

{% highlight python %}
# Create 100`000 samples drawn from a uniform distribution 
df = pd.DataFrame(columns = ("X","Y"), data = np.random.uniform((0,0),(1,1),(100000,2)))
df['Inside'] = df.apply(lambda row: ((row["X"] - 0.5)**2 + (row["Y"] - 0.5)**2) <= 0.5**2, axis=1)

piCalculated = (df['Inside'].sum() / df.shape[0]) / (0.5**2)

# Summarise results 
print(f"After {df.shape[0]} trials, {df['Inside'].sum()} samples fall within the circle.")
print(f"The estimated value of Pi is: {piCalculated}")

# Create a plot
plt.scatter(df[df["Inside"] == False]["X"], df[df["Inside"] == False]["Y"], color="b", marker=".")
plt.scatter(df[df["Inside"] == True]["X"], df[df["Inside"] == True]["Y"], color="r", marker=".")
plt.xlabel("X")
plt.ylabel("Y")
plt.title("Estimating Pi (100'000 Samples)")
plt.show()
{% endhighlight %}

![Using Monte Carlo method, 100'000 points are sampled. Points which are red are inside the circle, blue points fall outside the circle.](/assets/posts/monte-carlo-method-of-integration-2.png?raw=true){: width="50%"}

This is still only 3 digits accurate to $$ \pi $$, depending on rounding.

If we go further to 100 million samples, using a vectorized operation this time to improve performance, we start to get useful results.

{% highlight python %}
# Create 100`000`000 samples drawn from a uniform distribution 
df = pd.DataFrame(columns = ("X","Y"), data = np.random.uniform((0,0),(1,1),(100000000,2)))

# Using vector operations to improve performance
df['Inside'] = (df['X'] * df['X'] - 2*0.5*df['X'] + 0.5*0.5 + df['Y'] * df['Y'] - 2*0.5*df['Y'] + 0.5**2) <= 0.5**2

piCalculated = (df['Inside'].sum() / df.shape[0]) / (0.5**2)

# Summarise results 
print(f"After {df.shape[0]} trials, {df['Inside'].sum()} samples fall within the circle.")
print(f"The estimated value of Pi is: {piCalculated}")
{% endhighlight %}

![Using Monte Carlo method to calculate $$\pi$$ with 100'000'000 samples.](/assets/posts/monte-carlo-method-of-integration-3.png?raw=true){: width="50%"}

Now we have 5 digits of Pi, although it required a significant number of operations to achieve. Later I will discuss a method which could be used to improve on this. 

# The Monte-Carlo Estimator

We can apply the same technique to numerically calculate the integral of some function. 

To calculate the definite integral we will use the Monte Carlo estimator:

$$ \hat{F} = (b-a)\frac{1}{N}\sum^{N-1}_{i=0}f(X_i) $$

The Monte Carlo estimator is a function which approximates the value of an integral. We can show that the expectation of the Monte-Carlo estimator is the integral of the function. 

Taking the expectation of the Monte Carlo estimator:

$$ 
\begin{align}
E[\hat{F}] &= E \left[ (b-a)\frac{1}{N}\sum^{N-1}_{i=0} f(X_i) \right] \\
&= (b-a)\frac{1}{N}\sum^{N-1}_{i=0}E \left[ f(X_i) \right] 
\end{align}
$$

To find $$ E \left[ f(X) \right] $$ we can use the formula for the expected value of a continuous function of a random variable:

$$ E[g(X)] = \int^{X_{max}}_{X_{min}}{f(X)f_X(X)}dX $$

Here, $$X$$ is a random variable which in our case is a sample point, and $$f(X)$$ is the function of a random variable we want to integrate which is defined over the closed interval $$[X_{min}, X_{max}]$$. $$f_X(X)$$ is the probability density function (PDF) of the distribution from which the sample $$X$$ is drawn. We make no assumptions about how the samples should be drawn within the domain of $$f(.)$$ and use a uniform distribution with all possible sample points having equal probability. 

In one dimension, the PDF of the uniform distribution for the interval $$[a,b]$$ is given by:

$$
f_X(x) = 
\begin{cases} 
    \frac{1}{b-a} & a \leq x \leq b \\
    0 & x \lt a \text{ or } x \gt b 
\end{cases}
$$

Using the formula for the expectation of a continuous function, the expectation of the Monte-Carlo estimator is:

$$ 
\begin{aligned}
E[\hat{F}] &= (b-a)\frac{1}{N}\sum^{N-1}_{i=0} \int^{b}_{a} f(X)  \frac{1}{b-a} \, dX \\
&= \frac{1}{N}\sum^{N-1}_{i=0} \int^{b}_{a}{f(X)} \, dX \\
& = \int^{b}_{a}{f(X)} \, dX 
\end{aligned}        
$$

Which is the definite integral of $$f(X)$$.  

# Standard Error of the Monte-Carlo Estimator

We can make an estimate the spread in the result of the Monte-Carlo estimator as a function of the number of points, $$N$$, used in the calculation.

Recall the Monte-Carlo estimator described previously:

$$
\hat{F}= (b-a) \frac{1}{N}\sum^{N}_{i=1}f(X_i)
$$

Some properties of variances are:

* $$Var(cX) = c^2Var(X)$$ &nbsp;
* $$Var(X_1 + X_2 + ... + X_N) = Var(X_1) + Var(X_2) + ... + Var(X_N)$$ &nbsp;

We can use these properties to find the variance of $$\hat{F}$$:

$$
\begin{aligned}
Var(\hat{F}) &= \frac{(b-a)^2}{N^2}Var \left( f(X_1) + f(X_2) + ... + f(X_N) \right) \\
&= \frac{(b-a)^2}{N^2}\sum^{N}_{i=1}Var \left( f(X_i) \right)
\end{aligned}
$$

Since we expect $$ Var(X_i) = Var(X_{i+1})$$ as X is independant and identically distributed (i.i.d.), we can simplify this to:

$$
Var(\hat{F}) = \frac{(b-a)^2}{N}Var \left( f(X) \right)
$$

We can denote $$Var \left( f(X) \right) $$ as the population variance of $$ f(X) $$ using the notation $$\sigma^2_f$$:

$$
\sigma^2_{\hat{F}} = \frac{(b-a)^2}{N} \sigma^2_f
$$

Then the standard error of the Monte-Carlo estimator is found by taking the square root of the variance:

$$
\sigma_{\hat{F}} = (b-a)\sqrt{\frac{\sigma^2_f}{N}} 
$$

In practice, for estimating the standard error we would use a sample variance for $$ f(X) $$ calculated in evaluating the estimate of the integral, which we could call $$s^2_f$$.

$$
\sigma_{\hat{F}} = (b-a)\sqrt{\frac{s^2_f}{N}} 
$$

Where the sample variance is calculated:

$$
s^2_f = \frac{1}{N-1} \sum^{N}_{i=1}{(X_i-\bar{X})^2} 
$$

# Using the Monte-Carlo Estimator to Calculate Definite Integrals

For this example, the function $$ f(x)=x^2 $$ will be used. This function is trivial to integrate analytically and has the solution $$ F(x)=\frac{1}{3}x^3 $$, which we can use in comparison with the numerical result.

The definte integral will be evaluated over the interval $$[a,b]$$:

$$
\int_a^b{x^2}dx
$$

In the following Python example the function f() is defined which returns the value of $$ x^2 $$. The Monte Carlo estimator will be evaluated for the definite integral over the region [0, 100] using 1'000 samples. Additionally the function F() is defined as the analytical solution used for comparison. 

{% highlight python %}
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

# Function to integrate
def f(x):
    return x**2

# Definite integral of the function for comparison
def F(domain):
    return (domain[1]**3)/3 - (domain[0]**3)/3

# Domain over which to evaluate integral
domain = (0, 100)

# Create n samples drawn from a uniform distribution 
n = 1000000
df = pd.DataFrame(columns = ("X",), data = np.random.uniform(domain[0], domain[1], n))

{% endhighlight %}

The function is applied to the X column in the data frame and the return values stored in the Y column. The Monte Carlo estimate of the integral is calculated for an increasing size batch of points and the resulting values stored in lists for plotting.

{% highlight python %}

# Calculate function values using numpy vectorize to increase speed 
df['Y'] = np.vectorize(f)(df["X"])

# Empty list for errors
iterations = []
values = []
errors = []

# For increasing number of samples
step = 10000
for i in range(step, n+step, step):

    # Find the mean value of the function
    meanValue = df[0:i]['Y'].mean()
    
    # Find the variance of the function
    varianceValue = df[0:i]['Y'].var()

    # Find the error
    error = np.sqrt(varianceValue / i) * (domain[1] - domain[0])
    
    # Find the integral calculated over the domain
    calculatedIntegral = meanValue * (domain[1] - domain[0])

    # Save values
    values.append(calculatedIntegral)
    errors.append(error)
    iterations.append(i)
{% endhighlight %}

Printing and plotting the results:

{% highlight python %}

# Print the results 
print(f"Iterations: {iterations[-1]}")
print(f"Estimated error: {errors[-1]}")
print(f"Calculated integral: {values[-1]}")
print(f"Analytical solution: {F(domain)}")
print(f"Percent error is: {(100*(values[-1]-F(domain))/F(domain)):.4f} %")    
    
# Plot the integral as a function of points
plt.plot(iterations, values)
plt.xlabel("Samples")
plt.ylabel("F(.)")
plt.title("Monte Carlo Integration of $x^2$")
plt.show()
{% endhighlight %}

![Plot of calculated value (blue line) converging to true value (red dashed line).](/assets/posts/monte-carlo-method-of-integration-4.png?raw=true){: width="50%"}

In the following example we apply the same method to a more difficult integral to solve, which is $$ f(x) = \int^2_1{\mathrm{e}^{-\sin(x^2)}} \, dx $$.

{% highlight python %}
# Domain over which to evaluate integral
domain = (1, 2)

# Function to integrate
def f(x):
    return np.exp(-np.sin(x**2))

# Create n samples drawn from a uniform distribution 
n = 5000000
df = pd.DataFrame(columns = ("X",), data = np.random.uniform(domain[0], domain[1], n))
# User numpy vectorize rather than pandas apply to increase speed 
df['Y'] = np.vectorize(f)(df["X"])

# Empty list for errors
iterations = []
errors = []

print("Samples" + " " * 3 + "Monte Carlo Integral")
print("-" * 60)
step = 50000
for i in range(step, n, step):

    # Find the mean value of the function
    meanValue = df[0:i]['Y'].sum() / i

    # Find the integral calculated over the domain
    calculatedIntegral = meanValue * (domain[1] - domain[0])

    errors.append(calculatedIntegral)
    iterations.append(i)
    
    # Display results
    print(f"{i}     {calculatedIntegral}")
    
plt.plot(iterations,errors)
plt.xlabel("Samples")
plt.ylabel("F(.)")
plt.title("Monte Carlo Integration of f(.)")
plt.show()
{% endhighlight %}

The calculated value at the end of the iterations is 0.7248, which is close to a value we might expect from a numerical solution of this definite integral. 

![Plot of calculated value vs. number of samples.](/assets/posts/monte-carlo-method-of-integration-5.png?raw=true){: width="50%"}

# Using the Monte-Carlo Estimator to Calculate Double Integrals

For this example, we will integrate the function $$f(x,y)=\left(x^2+y^2\right)^2$$ over a region $$D$$ defined by a circle of unit radius centered about the origin.

$$
I = \int\int_{D} (x^2+y^2)^2 \, dx \, dy
$$

The visualisation created in Python below shows the surface of the function to be integrated.

{% highlight python %}

def f(theta, r):
    """ The function to integrate. """
    # Convert to rectangular coordinates
    x = r*np.cos(theta)
    y = r*np.sin(theta)
    
    # Evaluate result
    return (x**2 + y**2)**2 

# Create grid of values to apply the function to
theta, r = np.meshgrid(np.arange(-0, 2*np.pi, 0.05), np.arange(-0, 1, 0.05))

# Apply function
fValues = np.array(f(np.ravel(theta), np.ravel(r)))
z = fValues.reshape(theta.shape)

# Transform to rectangular coordinates
xTransformed = r*np.cos(theta)
yTransformed = r*np.sin(theta)

# Plot surface
ax = Axes3D(plt.gcf())
ax.plot_surface(xTransformed, yTransformed, z)
ax.set_title("Surface of $(x^2+y^2)^2$ in Unit Circle")
ax.set_xlabel("x axis")
ax.set_ylabel("y axis")
ax.set_zlabel("$(x^2+y^2)^2$")
plt.show()

{% endhighlight %}

![Paraboloid surface of the function to be integrated is steepest closer to the radius of the unit circel.](/assets/posts/monte-carlo-method-of-integration-8.png?raw=true){: width="50%"}

We can find an analytical solution to this problem by changing to polar coordinates, where $$\theta$$ is angle and $$r$$ is the magnitude of the radius. In doing so the Pythagorean theorem, $$r^2 = x^2 + y^2$$, is used.

$$
\begin{align}
I &= \int_0^{1}\int_{0}^{2\pi} r^4 \, r \, d\theta \, dr \\
  &= \int_0^{1} r^5 \left[\theta + C\right]_{\theta=0}^{\theta=2\pi} \, dr \\
  &= 2\pi\int_0^{1} r^5  \, dr \\
  &= 2\pi \left[ \frac{1}{6}r^6 + C \right]_{r=0}^{r=1} \\ 
  &= \frac{1}{3}\pi \\ 
\end{align}
$$

The analytical solution will be used to compare with the numerical solution calculated in Python.

In rectangular coordinates, the function to be integrated is defined:

{% highlight python %}

# The function to integrate
def f(x, y):
    return x**2 + y**2

{% endhighlight %}

A more general version of the Monte-Carlo estimator is:

$$
\hat{F}= V \frac{1}{N}\sum^{N}_{i=1}f(X_i)
$$

Where the variable V will be the volume of the region of integration, which in this case is the surface area of the unit circle we wish to integrate over:

{% highlight python %}
# The volume over which we will integrate is the area of the unit circle
V = np.pi
{% endhighlight %}

This example will use 1'000'000 two-dimensional samples over a square grid enclosing the unit circle, which are sampled from a uniform distribution:  

{% highlight python %}
n = 1000000
df = pd.DataFrame(
    columns = ("X", "Y"), 
    data = np.random.uniform((-1,-1), (1,1), (n, 2)))
{% endhighlight %}

Selecting only the points within the unit circle:

{% highlight python %}
df = df[(df["X"]**2 + df["Y"]**2) <= 1]
{% endhighlight %}

Applying the function to be integrated to the points as a vectorized operation: 

{% highlight python %}
# User numpy vectorize rather than pandas apply to increase speed 
df["Z"] = np.vectorize(f)(df["X"], df["Y"])
{% endhighlight %}

Next, the integral and estimated error for several batches of points increasing in size is calculated:

{% highlight python %}
# Calculate the integral for an increasing number of points
print(" Samples" + " " * 5 + "Integral" + " " * 3 + "Error Est.")
print("-" * 35)
iterations = []
errors = []
step = 5000
for i in range(step, n+step, step):

    # Calculate the estimated error
    varianceValue = df[0:i]['Z'].var()
    error = np.sqrt(varianceValue / i) * V
    iterations.append(i)
    
    # Calculate the integral over the domain
    meanValue = df[0:i]['Z'].mean()
    calculatedIntegral = meanValue * V
    errors.append(calculatedIntegral)
    
    # Display results
    print(f"{i:8d}     {calculatedIntegral:8f}   {error:8f}")
{% endhighlight %}

Plotting the results:

{% highlight python %}
# Print the results 
print(f"Iterations: {iterations[-1]}")
print(f"Estimated error: {errors[-1]}")
print(f"Calculated integral: {values[-1]}")
print(f"Analytical solution: {np.pi/2}")
    
# Plot the integral as a function of points
plt.plot(iterations, values)
plt.xlabel("Samples")
plt.ylabel("F(.)")
plt.title("Monte Carlo Integration of f(.)")
plt.show()
{% endhighlight %}

In the plot below the Monte-Carlo estimate has reached 1.0476 for this sample size, which is very close to the analytical solution of 1.0472.  

![Value of the double integral converging with an increasing number of samples.](/assets/posts/monte-carlo-method-of-integration-6.png?raw=true){: width="50%"}

# Recursive Stratified Sampling

In stratified sampling we wish to distribute the number of sample points available over the region of integration such that areas with higher variance are allocated a higher number of samples. This results in greater efficiency in the use of samples by obtaining a better estimate of the mean in areas where there is more detail. The MISER algorithm [[1]](#1) is an example of a recursive stratified sampling implementation. The stratified sampling algorithm presented here is based on the MISER algorithm however has some simplifications which will be explained.

The code block below shows the implementation of the recursive stratified sampling algorithm.

{% highlight python %}

from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import pandas as pd

def divide_annulus(t):
    """ Finds radius which divides an annulus into two equal volumes.  """
    return np.sqrt((t[1]**2)/2 + (t[0]**2)/2)

def estimate_polar(f, N, boundTheta, boundR, V):
    """ Monte-Carlo estimate in polar coordinates. """
    # Generate sample points
    dfPoints = pd.DataFrame(
        columns = ("theta", "r"), 
        data = np.random.uniform((boundTheta[0],boundR[0]**2), (boundTheta[1],boundR[1]**2), (N, 2)))

    # Transform R for evenly distributed density of points along radius
    dfPoints["r"] = np.sqrt(dfPoints["r"])

    # Use numpy vectorize calculation of function value 
    dfPoints["value"] = np.vectorize(f)(dfPoints["theta"], dfPoints["r"])

    return dfPoints["value"].mean()*V \
            ,(V**2)*dfPoints["value"].var()/N \
            ,dfPoints
        
def rss_polar(f, N, k, boundTheta, boundR, maxError, maxDepth, depth=1):
    """ Recursive stratified sampling implementation. """
    
    # Calculate the volume of this sector of the annulus
    V = (boundR[1]**2)*(abs(boundTheta[1] - boundTheta[0])/2) - (boundR[0]**2)*(abs(boundTheta[1] - boundTheta[0])/2)
    
    # If termination criteria not met yet
    if(depth < maxDepth
      and N > 4):
        
        # Sample for estimate of variance
        dfSampleTheta1 = pd.DataFrame(
            columns = ("theta", "r"), 
            data = np.random.uniform((boundTheta[0],boundR[0]), (np.mean(boundTheta),boundR[1]), (k, 2)))
        dfSampleTheta2 = pd.DataFrame(
            columns = ("theta", "r"), 
            data = np.random.uniform((np.mean(boundTheta),boundR[0]), (boundTheta[1],boundR[1]), (k, 2)))
        dfSampleR1 = pd.DataFrame(
            columns = ("theta", "r"), 
            data = np.random.uniform((boundTheta[0],boundR[0]), (boundTheta[1],divide_annulus(boundR)), (k, 2)))
        dfSampleR2 = pd.DataFrame(
            columns = ("theta", "r"), 
            data = np.random.uniform((boundTheta[0],divide_annulus(boundR)), (boundTheta[1],boundR[1]), (k, 2)))
        
        # Calculate values
        dfSampleTheta1["value"] = np.vectorize(f)(dfSampleTheta1["theta"], dfSampleTheta1["r"])
        dfSampleTheta2["value"] = np.vectorize(f)(dfSampleTheta2["theta"], dfSampleTheta2["r"])
        dfSampleR1["value"] = np.vectorize(f)(dfSampleR1["theta"], dfSampleR1["r"])
        dfSampleR2["value"] = np.vectorize(f)(dfSampleR2["theta"], dfSampleR2["r"])
        
        # Calculate the combined standard error of the samples
        sample = pd.concat([dfSampleTheta1, dfSampleTheta2, dfSampleR1, dfSampleR2], axis=0)
        sampleVariance = sample["value"].var()
        errorSample = np.sqrt(sampleVariance / sample.shape[0]) * V
        
        # Error is a termination criteria
        if(errorSample <= maxError):
            valueFinal, errorFinal, dfPointsFinal = estimate_polar(f, N, boundTheta, boundR, V)
        else:    
            # Calculate variances
            dfSampleTheta1Variance = dfSampleTheta1["value"].var()
            dfSampleTheta2Variance = dfSampleTheta2["value"].var()
            dfSampleR1Variance = dfSampleR1["value"].var()
            dfSampleR2Variance = dfSampleR2["value"].var()
            
            # Choose the dimension to split on such that the sum of standard errors in two dimensions is reduced 
            if((np.sqrt(dfSampleTheta1Variance) + np.sqrt(dfSampleTheta2Variance)) 
               < (np.sqrt(dfSampleR1Variance) + np.sqrt(dfSampleR2Variance))):
                # Split along Theta direction
                Na = int(N*np.sqrt(dfSampleTheta1Variance)/(np.sqrt(dfSampleTheta1Variance)+np.sqrt(dfSampleTheta2Variance)))
                Na = max(2, min(N-2, Na))
                Nb = N - Na
                valueSub1, errorSub1, dfPointsSub1 = rss_polar(f, Na, k, (boundTheta[0], np.mean(boundTheta)), boundR, maxError, maxDepth, depth+1)
                valueSub2, errorSub2, dfPointsSub2 = rss_polar(f, Nb, k, (np.mean(boundTheta), boundTheta[1]), boundR, maxError, maxDepth, depth+1)  
            else:
                # Else split along R direction
                Na = int(N*np.sqrt(dfSampleR1Variance)/(np.sqrt(dfSampleR1Variance)+np.sqrt(dfSampleR2Variance)))
                Na = max(2, min(N-2, Na))
                Nb = N - Na
                valueSub1, errorSub1, dfPointsSub1 = rss_polar(f, Na, k, boundTheta, (boundR[0], divide_annulus(boundR)), maxError, maxDepth, depth+1)
                valueSub2, errorSub2, dfPointsSub2 = rss_polar(f, Nb, k, boundTheta, (divide_annulus(boundR), boundR[1]), maxError, maxDepth, depth+1)
                            
            # Final estimate
            meanSub1 = dfPointsSub1["value"].mean()
            meanSub2 = dfPointsSub2["value"].mean()
            errorFinal = errorSub1/4 + errorSub2/4
            valueFinal = valueSub1 + valueSub2
            dfPointsFinal = pd.concat([dfPointsSub1, dfPointsSub2], axis=0)

    # Otherwise terminate here with a Monte-Carlo estimate of remaining points    
    else: 
        valueFinal, errorFinal, dfPointsFinal = estimate_polar(N, boundTheta, boundR, V)

    # Return final values
    return valueFinal, errorFinal, dfPointsFinal
        
{% endhighlight %}

The estimate_polar() routine calculates the Monte-Carlo estimate for a function defined in polar coordinates over the sector of an annulus defined by bounds on angle $$\theta$$ and radius $$r$$. The rss_polar() routine recursively combines Monte-Carlo estimates up to a maximum depth and maximum error defined by maxError and maxDepth respectively. Each recursive call of the function returns an estimate of the integral for the region and it's standard error, together with a collection of points used in evaluating the sub-integral. In the implementation presented in this notebook an additional constraint is made which ensures $$N>=4$$ and both $$N_a\ge2$$ and $$N_b\ge2$$ on each call.

Each sub-region of the integration is a sector of an annulus defined by bounds on angle $$\theta_0$$, $$theta_1$$ and bounds on radius $$r_0$$, $$r_1$$. The following method is used to split the annulus sector radially into equal volumes:

$$
\sqrt{\frac{r_0^2}{2} + \frac{r_1^2}{2}}
$$

As the sub-regions of integration are of equal volume, the sub-integral values for sub-regions $$A$$ and $$B$$ are combined as the sum of integral values for the sub-regions: 

$$
I_{final} = I_A + I_B
$$

And the standard error is calculated:

$$
\sigma_{final} = \frac{1}{4}\sigma_A + \frac{1}{4}\sigma_B
$$

Like the MISER algorithm, this implementation recursively divides the region of integration along one dimension. The first problem which arises is how to choose which dimension to split the region of integration along. In this implementation the dimension is chosen such that sum of standard deviations in each of the candidate sub-regions is minimized. While this approach is discussed in [[1]](#1), the MISER algorithm itself selects the dimension according to the split which minimizes the sum of a power of the difference between minimum and maximum function values in each of the sub-regions.  

The second problem is how to allocate points between the sub-regions. In this implementation sub-regions are chosen such that they are of equal volume. The combined expectation of the function values from both sub-regions is then the mean of the expectations.

$$
\left<f\right> = \frac{1}{2}\left( \left<f\right>_a + \left<f\right>_b \right)
$$

We would like to know the variance of this overall expectation. Recalling that $$Var(cX) = c^2Var(X)$$, we find:

$$
Var(\left<f\right>) = \frac{1}{4}Var(\left<f\right>_a) + \frac{1}{4}Var(\left<f\right>_b)
$$

The variance of the sample mean is $$ Var(\bar{X}) = \frac{\sigma_X^2}{N} $$, so that the variance of the combined sample mean of the function may be expressed in terms of the variance of the function value in each sub-region:

$$
Var(f) = \frac{\sigma^2_a}{4N_a}+\frac{\sigma^2_b}{4N_b}
$$

Next, we would like to find $$N_a$$ and $$N_b$$ such that the variance is minimized:

$$
\text{argmin}_{N_a,N_b} \left( \frac{\sigma^2_a}{4N_a} + \frac{\sigma^2_b}{4N_b} \right)
$$

To simplify the problem we will make the substitution:

$$
\text{Let} \, x = \frac{N_a}{N}
$$

Then choose $$x$$ such that the combined variance is minimized:

$$
\text{argmin}_{x} \left( \frac{\sigma^2_a}{4xN} + \frac{\sigma^2_b}{4(N-xN)} \right)
$$

To do so we find the derivative of the expression:

$$
\frac{d}{dx} \left( \frac{\sigma_a^2}{4N}x^{-1} + \frac{\sigma_b^2}{4N-4Nx} \right) = \frac{4N\sigma_b^2}{(4N-4Nx)^2}-\frac{\sigma_a^2}{4Nx^2}
$$

Then set the derivative equal to zero to find the stationary points of the expression:

$$
\begin{align}
0&=16N^2\sigma_b^2x^2-\sigma_a^2(4N-4Nz)^2 \\
&= (\sigma_b^2-\sigma_a^2)x^2+2\sigma_a^2x-\sigma_a^2
\end{align}
$$

Using the quadratic formula we find the values of $$x$$:

$$
\begin{align}
x&=\frac{-2\sigma_a^2\pm\sqrt{(2\sigma_a^2)^2+4\sigma_a^2(\sigma_b^2-\sigma_a^2)}}{2(\sigma_b^2-\sigma_a^2)}\\
&=\frac{-2\sigma_a^2\pm\sqrt{4\sigma_b^2\sigma_a^2}}{2(\sigma_b^2-\sigma_a^2)}\\
&=\frac{-\sigma_a^2\pm\sigma_b\sigma_a}{(\sigma_b^2-\sigma_a^2)}
\end{align}
$$

Using the difference of two squares:

$$
\begin{align}
x &= -\frac{\sigma_a(\sigma_a\pm\sigma_b)}{(\sigma_b+\sigma_a)(\sigma_b-\sigma_a)} \\
&= \frac{\sigma_a}{\sigma_a-\sigma_b} \, \text{or} \, \frac{\sigma_a}{\sigma_a+\sigma_b}
\end{align}
$$

We choose $$\frac{\sigma_a}{\sigma_a+\sigma_b}$$ as this results in a value of $$N_a$$ which is positive and less than $$N$$:

$$
\frac{N_a}{N_a+N_b} = \frac{\sigma_a}{\sigma_a+\sigma_b}
$$

We can then calculate $$N_a$$ as:

$$
N_a = \frac{\sigma_a}{\sigma_a+\sigma_b}N
$$

And for $$N_b$$:

$$
N_b = N - N_a 
$$

This implementation also differs from the MISER algorithm as a constant number of sample points, $$k$$, is used in evaluating termination criteria and in determining the dimensional split, rather than a fraction of N as in [[1]](#1). A keyword argument, depth, is used by the recursive funciton to track recursive depth for termination criteria.

In the following Python example the recursive stratified sampling method introduced above is used to find the Monte-Carlo estimate of the double integral from the previous double integral example. The termination criteria are set with a minimum standard error of 0.0001 and maximum depth of 10 for recursion. 

{% highlight python %}

# Maximum number of sample points
N = 4400

# Unit circle
boundTheta = (0, 2*np.pi)
boundR = (0, 1)

# Termination Criteria
maxError = 0.0001
maxDepth = 10

# Sample points for evaluating dimensional split and termination criteria
k = 100

calculatedIntegral, calculatedError, df = rss_polar(f, N, k, boundTheta, boundR, maxError, maxDepth)
calculatedError = np.sqrt(calculatedError)

{% endhighlight %}

Plotting the points and printing the Monte-Carlo estimate:

{% highlight python %}

# Transform the data
df["x"] = df["r"]*np.cos(df["theta"])
df["y"] = df["r"]*np.sin(df["theta"])

plt.scatter(df["x"], df["y"], marker='.', alpha=0.2)
plt.xlabel("x")
plt.ylabel("y")
plt.title("Integration Points of $(X^2+y^2)^2$")
plt.show()

# Analytical solution of the integral
analyticalIntegral = np.pi/3

# Print the results 
print(f"Estimate standard error: {calculatedError}")
print(f"Calculated integral: {calculatedIntegral}")
print(f"Analytical solution: {analyticalIntegral}")
print(f"Percent error is: {(100*(calculatedIntegral-analyticalIntegral)/analyticalIntegral):.4f} %")

{% endhighlight %}

From the plot of sample points we can see that most points are distributed along the radius of the circle which is where the function is steepest. For 4'400 points we get estimates with a precentage error which is in the order of magnitude of 0.05%, considerably less sample points than what would be required for a similar effort when using a plain Monte-Carlo estimate.

![Value of the double integral converging with an increasing number of samples.](/assets/posts/monte-carlo-method-of-integration-7.png?raw=true){: width="50%"}

# Python Notebook and GitHub Repository

The Python notebook for this post is avilable in my GitHub repository [located here](https://github.com/bott-j/monte-carlo-integration-python)

## References

<a id="1">[1]</a> 
W. H. Press and G. R. Farrar, "Recursive Stratified Sampling for Multidimensional Monte Carlo Integration," in Computers in Physics. vol. 4, 1990, pp. 190–195.