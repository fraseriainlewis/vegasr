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
lower = np.array([-0.5, -0.5, -0.5])
upper = np.array([0.5, 0.5, 0.5])

# Define the integrand
# vegas expects a function that takes a (n, dim) array and returns a (n,) array
mvn = multivariate_normal(mean=mu, cov=cov)

print(mvn.pdf(np.array([0.1,0.1,0.1])))

@vegas.lbatchintegrand
def f(x,y,z):
  print(x.shape)
  return mvn.pdf(x)

@vegas.lbatchintegrand
class vegasHelper:
    def __init__(self, y, z):
        self.y = y
        self.z = z
      
    def __call__(self, theta):
        return(f(theta,self.y,self.z))

y=1.0
z=1.0

newf = vegasHelper(y=y,z=z)

# Initialize the integrator
integ = vegas.Integrator([[l, u] for l, u in zip(lower, upper)])
# Adaptation phase
integ(newf, nitn=10, neval=1000)
# Final integration
result = integ(newf, nitn=10, neval=1000)
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





