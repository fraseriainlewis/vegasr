
#explicitly set seed
gvar.ranseed(99990)
integ = vegas.Integrator(mx.array([[-1, 1],[-1,1],[-1, 1],[-1,1],[-1,1],[-1,1],[0.0001,1],[0.0001,1]],dtype=mx.float32))
#integ.map.grid.base[0,0:5] # get the actual grid typically 1000 bins

res_warmup = integ(model, nitn=50, neval=100000)
print(res_warmup.summary())

result = integ(model, nitn=1, neval=100000,adapt=False)
print(result.summary())

# Production (Stable Integration)
#result = integ(model, nitn=10, neval=100000)
#log_evidence = model.l_max + mx.log(gvar.mean(result))
print(f"Log Evidence: {log_evidence}")

# Step 2: Draw samples and weights from the adapted grid                                                              

x_samples = []                                                                                                      
raw_weights = []                                                                                                          
i=1
#gvar.ranseed(99990)
#integ.random_batch(): - faster but does in batches of x and wgt
for x, wgt in integ.random_batch():
  #print(x)
  print(i)
  i=i+1
  print(len(x))
  myx = x 
  f_vals=mx.exp(compute_log_lik(mx.transpose(mx.array(myx)), model.y,model.T,model.K)-model.l_max) #np.array([xi for xi in x])   # or vectorized
  #importance weight ∝ f(x) * wgt  (wgt = jac / neval from the grid)
  mywgt=wgt
  raw_weights.append(f_vals * wgt)
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

