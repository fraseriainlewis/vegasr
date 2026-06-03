
#explicitly set seed
gvar.ranseed(99990)
integ = vegas.Integrator(mx.array([[-1, 1],[-1,1],[-1, 1],[-1,1],[-1,1],[-1,1],[0.0001,1],[0.0001,1]],dtype=mx.float32))
#integ.map.grid.base[0,0:5] # get the actual grid typically 1000 bins

res_warmup = integ(model, nitn=10, neval=100000)
print(res_warmup.summary())

result = integ(model, nitn=1, neval=60000,adapt=False)
print(result.summary())

# Production (Stable Integration)
#result = integ(model, nitn=10, neval=100000)
#log_evidence = model.l_max + mx.log(gvar.mean(result))
print(f"Log Evidence: {log_evidence}")

# Step 2: Draw samples and weights from the adapted grid                                                              

x_samples = []                                                                                                      
raw_weights = []                                                                                                          
#i=1
#gvar.ranseed(99990)
#integ.random_batch(): - faster but does in batches of x and wgt
for x, wgt in integ2.random_batch():
  #print(x)
  #print(i)
  #i=i+1
  #print(len(x))
  #myx = x 
  #f_vals=mx.exp(compute_log_lik(mx.transpose(mx.array(myx)), model.y,model.T,model.K)-model.l_max) #np.array([xi for xi in x])   # or vectorized
  #importance weight ∝ f(x) * wgt  (wgt = jac / neval from the grid)
  #mywgt=wgt
  raw_weights.append(wgt)
  x_samples.append(x.copy())   

x_all = np.concatenate(x_samples, axis=0)
w_all = np.concatenate(raw_weights)
# Step 3: Normalize weights
w_all = np.clip(w_all, 0, None)
w_norm = w_all / w_all.sum()

# Step 4: Effective sample size
ess = 1.0 / np.sum(w_norm**2)
print(f"ESS: {ess:.0f}")

# Step 5: Resample to get unweighted draws
idx = np.random.choice(len(x_all), size=2000, replace=True, p=w_norm)
posterior_samples = x_all[idx]

gvar.ranseed(99990)
np.random.seed(42)
amap = integ2.map
n_samples = 10
ndim = 3
# Sample uniformly in y-space (unit hypercube)
y = np.random.uniform(0, 1, (n_samples, ndim))
x = np.empty_like(y)
jac = np.empty(n_samples)
# Map y → x, filling jac = dx/dy (product over dimensions)
amap.map(y, x, jac)
jac2 = integ2.map.jac(y)
print(jac)
print(jac2)


ninc = amap.ninc
inc  = amap.inc
ndim = amap.dim
  
jac3 = 1.0#np.ones(y.shape[0])
for d in range(ndim):
  k = np.floor(y[:, d] * ninc[d]).astype(int)
  k = np.clip(k, 0, np.array(ninc[d]) - 1)
  jac3 *= ninc[d] * np.array(inc)[d, k]
print(jac3)



inc = integ2.map.inc
grid_list = integ2.map.extract_grid()

# Importance weight: p_vegas(x) = 1/jac, so w ∝ f(x) * jac
f_vals=mx.exp(compute_log_lik(mx.transpose(mx.array(x)), model.y,model.T,model.K)-model.l_max)
weights = f_vals * jac
weights = np.clip(weights, 0, None)
weights /= weights.sum()
# Step 4: Effective sample size
ess = 1.0 / np.sum(weights**2)
print(f"ESS: {ess:.0f} {100*ess/n_samples}")

grid_list = integ2.map.extract_grid()
# Bin widths (= dx/dy proportional, the Jacobian per bin per dim)
inc = integ2.map.inc # shape: (ndim, ninc) # jacob is  ninc * inc where ninc is size of grid





compute_jac(y,amap)

ninc = amap.ninc
inc  = amap.inc
ndim = amap.dim
  
jac = 1.0#np.ones(y.shape[0])
for d in range(ndim):
  k = np.floor(y[:, d] * ninc[d]).astype(int)
  k = np.clip(k, 0, np.array(ninc[d]) - 1)
  jac *= ninc[d] * np.array(inc)[d, k]


def compute_jac(y, amap):
  ninc = amap.ninc
  inc  = amap.inc
  ndim = amap.dim
  
  jac = np.ones(y.shape[0])
  for d in range(ndim):
    k = np.floor(y[:, d] * ninc).astype(int)
    k = np.clip(k, 0, np.array(ninc) - 1)
    jac *= ninc * np.array(inc)[d, k]
  return jac




