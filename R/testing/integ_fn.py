#### first part is all python, second part uses R function
import vegas
import numpy as np
from scipy.stats import multivariate_normal

# Parameters for MVN
mu = np.array([0.5, -0.2, 0.1])
cov = np.array([
        [1.0, 0.5, 0.2],
        [0.5, 1.2, 0.3],
        [0.2, 0.3, 0.8]
    ])

mvn = multivariate_normal(mean=mu, cov=cov)

print(mvn.pdf(np.array([0.1,0.1,0.1])))

# Integration limits
lower = np.array([-5.0, -5.0, -5.0])
upper = np.array([5.0, 5.0, 5.0])

# Define the integrand
# vegas expects a function that takes a (n, dim) array and returns a (n,) array
mvn = multivariate_normal(mean=mu, cov=cov)

print(mvn.pdf(np.array([0.1,0.1,0.1])))

@vegas.lbatchintegrand
def f(x):
  print(x.shape)
  return mvn.pdf(x)

# Initialize the integrator
integ = vegas.Integrator([[l, u] for l, u in zip(lower, upper)])
# Adaptation phase
integ(f, nitn=10, neval=1000)
# Final integration
result = integ(f, nitn=10, neval=1000)
print(result.summary())

#### now use R function
@vegas.lbatchintegrand
def ff(x):
  print(x.shape)
  return r.myf(x)
  #return r_func(np.transpose(x))

a=np.array([[0.1,0.1,0.1],[0.1,0.1,0.1]])

r.myf(a)
r_func(a)
#print(r.myf(np.array([[0.1,0.1,0.1],[0.1,0.1,0.1]])))
print(r_func(np.transpose(np.array([[0.1,0.1,0.1],[0.1,0.1,0.1]]))))

# Initialize the integrator
integ2 = vegas.Integrator([[l, u] for l, u in zip(lower, upper)])
# Adaptation phase
integ2(ff, nitn=10, neval=1000)
# Final integration
result2 = integ2(ff, nitn=10, neval=1000)
print(result2.summary())





