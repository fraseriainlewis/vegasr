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

