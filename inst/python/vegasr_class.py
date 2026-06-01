import vegas
import numpy as np
import json
import math
import gvar

class vegasr_wrapper:
    def __init__(self):
        self.integresults = []
        self.success = True

    def clear_results(self):
        self.integresults.clear()
    
    def add_results(self, integ):
        self.integresults.append(integ)

    def get_final_wt_results(self):
        res=vegas.ravg(self.integresults)
        #return(np.array([res.mean,res.sdev])) # if r_func() return array this does not work
        return np.array([gvar.mean(res), gvar.sdev(res)])  # works for both RAvg and RAvgArray
        # Note - this does not reach back into individual iterations but weights each summary
        # this is mathematically correct but Lepage notes can introduce tiny bias and if neval is very small using
        # a frozen grid might be appropriate. See python vegas website. 
    
    def get_x_wgts(self,integ):
      x_samples = []
      raw_weights = []
      for x, wgt in integ.random_batch():
        raw_weights.append(wgt) # needs * f(x), (wgt = jac / nincs from the grid)
        x_samples.append(x.copy())
      x_all = np.concatenate(x_samples, axis=0) # each x point in integrand f(x)
      w_all = np.concatenate(raw_weights) # weight in p(x) not f(x), needs multiplied by f(x)
      return([x_all,w_all])
    
    def get_grid(self,integ):
      return(integ.map.extract_grid())
      # actual bins
      
    def get_grid_inc(self,integ):
      return(np.array(integ.map.inc))
      # Bin widths (= dx/dy proportional, the Jacobian per bin per dim)
     
    
    def get_all_wt_results(self):
        return(self.integresults)




    def create_integrator(self, bounds):
        itg = vegas.Integrator(bounds)
        itg_id = str(len(self.integrators))
        self.integrators[itg_id] = itg
        return itg_id

    def integrate(self, itg_id, f, nitn=10, neval=1000, **kwargs):
        itg = self.integrators[itg_id]
        result = itg(f, nitn=nitn, neval=neval, **kwargs)
        
        # Format results for R consumption
        return {
            "mean": float(result.mean),
            "sdev": float(result.sdev),
            "chi2": float(result.chi2),
            "q": float(result.Q),
            "itn_results": [{"mean": float(r.mean), "sdev": float(r.sdev)} for r in result.itn_results]
        }

