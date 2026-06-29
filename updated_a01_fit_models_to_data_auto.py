#!/bin/env python

"""
Python script for estimation of AP model parameters to fit optically measured rabbit ventricular AP traces.
Automatically pairs before/after drug traces.

Version date:
   2023-11-22, Bearsden

Author:
   Zhechao Yang email:2712549y@student.gla.ac.uk

Usage:
  a01_fit_models_to_data_auto.py [-h] [options]

Options:
  -h, --help                   Show this screen.
  --fit_data                   Fit experimental data.
  --data_path=<path>           Path to AP traces (.csv or .csa) [default: ./data/]
  --output_file=<output_file>  CSV output with estimated parameters [default: EstimatedParVals_210421_run8_9_29_CaLblock.csv]
  --sse                        Minimise SumOfSquaresError [default: False]
  --rmse                       Minimise RootMeanSquaredError [default: False]
  --likelihood                 Maximise likelihood [default: False]
"""

from docopt import docopt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os, glob, re, csv
import myokit
import pints
from scipy.interpolate import interp1d
import scipy.stats as ss

np.set_printoptions(threshold=np.inf, linewidth=np.inf)

# Parameters to fit
all_to_fit = ['NaCa', 'NaK', 'Clb', 'CaL', 'tos', 'K1', 'Ks', 'Kr', 'Krblock', 'CaLblock']
parms_to_fit = ['I' + p for p in all_to_fit]




def jump_times(file1, file2):
    t1, y1, jump_times1 = get_experimental_trace(file1)
    t2, y2, jump_times2 = get_experimental_trace(file2)
    
    return jump_times1, jump_times2



class ActionPotentialModel(pints.ForwardModel):
    def __init__(self, jump_time1=0, jump_time2=0, duration=5.0, period = 600.0):
        super().__init__()
        
        self.jump_time1 = jump_time1
        self.jump_time2 = jump_time2
        self.duration   = duration
        self.period     = period
        
        self.parms = parms_to_fit
        self.parms1 = self.parms[:-2]
        self.parms2 = self.parms
    

    def simulate(self, parameters, times):
        split_index = np.argmax(np.diff(times) > 2 * (times[1] - times[0]))
        times1 = times[:split_index + 1]
        times2 = times[split_index + 1:]

        m1, _, _ = myokit.load("shannon_wang_puglisi_weber_bers_2004_a_p.mmt")
        m1.var('reversal_potentials.E_Na_SL').set_rhs(-15.0)
        p1 = myokit.Protocol()
        p1.schedule(level=-1.0, start=self.jump_time1-self.duration, duration=self.duration, period=self.period, multiplier=0)
        
        for i, par in enumerate(self.parms1):
            m1.var(par + '.p').set_rhs(parameters[i])
        s1 = myokit.Simulation(m1, p1)
        d1 = s1.run(times1[-1] + 0.01, log_times=times1)
        v1 = np.asarray(d1['cell.V'])

        m2, _, _ = myokit.load("shannon_wang_puglisi_weber_bers_2004_a_p.mmt")
        m2.var('reversal_potentials.E_Na_SL').set_rhs(-15.0)
        p2 = myokit.Protocol()
        p2.schedule(level=-1.0, start=self.jump_time2-self.duration, duration=self.duration, period=self.period, multiplier=0)
        
        for i, par in enumerate(self.parms2):
            m2.var(par + '.p').set_rhs(parameters[i])
        s2 = myokit.Simulation(m2, p2)
        dt = times1[1] - times1[0]
        d2 = s2.run(times2[-1] - times1[-1] - 10*dt + 0.01, log_times=times2 - times1[-1] - 10*dt)
        v2 = np.asarray(d2['cell.V'])

        return np.concatenate([v1, v2])
       
    def simulate_quantity(self, parameters, times):
        split_index = np.argmax(np.diff(times) > 2 * (times[1] - times[0]))
        times1 = times[:split_index + 1]
        times2 = times[split_index + 1:]

        m1, _, _ = myokit.load("shannon_wang_puglisi_weber_bers_2004_a_p.mmt")
        m1.var('reversal_potentials.E_Na_SL').set_rhs(-15.0)
        p1 = myokit.Protocol()
        p1.schedule(level=-1.0, start=self.jump_time1-self.duration, duration=self.duration, period=self.period, multiplier=0)
        
        for i, par in enumerate(self.parms1):
            m1.var(par + '.p').set_rhs(parameters[i])
        s1 = myokit.Simulation(m1, p1)
        d1 = s1.run(times1[-1] + 0.01, log_times=times1)
        v1 = np.asarray(d1['cell.V'])
        t1 = np.asarray(d1['environment.time'])

        m2, _, _ = myokit.load("shannon_wang_puglisi_weber_bers_2004_a_p.mmt")
        m2.var('reversal_potentials.E_Na_SL').set_rhs(-15.0)
        p2 = myokit.Protocol()
        p2.schedule(level=-1.0, start=self.jump_time2-self.duration, duration=self.duration, period=self.period, multiplier=0)
        
        
        for i, par in enumerate(self.parms2):
            m2.var(par + '.p').set_rhs(parameters[i])
        s2 = myokit.Simulation(m2, p2)
        dt = times1[1] - times1[0]
        d2 = s2.run(times2[-1] - times1[-1] - 10*dt + 0.01, log_times=times2 - times1[-1] - 10*dt)
        v2 = np.asarray(d2['cell.V'])
        t2 = np.asarray(d2['environment.time'])
        
        return t1,v1,t2,v2   
    
    
    def simulate_datalog(self, parameters, times):
        split_index = np.argmax(np.diff(times) > 2 * (times[1] - times[0]))
        times1 = times[:split_index + 1]
        times2 = times[split_index + 1:]

        m1, _, _ = myokit.load("shannon_wang_puglisi_weber_bers_2004_a_p.mmt")
        m1.var('reversal_potentials.E_Na_SL').set_rhs(-15.0)
        p1 = myokit.Protocol()
        p1.schedule(level=-1.0, start=self.jump_time1-self.duration, duration=self.duration, period=self.period, multiplier=0)
        
        
        for i, par in enumerate(self.parms1):
            m1.var(par + '.p').set_rhs(parameters[i])
        s1 = myokit.Simulation(m1, p1)
        d1 = s1.run(times1[-1] + 0.01, log_times=times1)
        v1 = np.asarray(d1['cell.V'])
        t1 = np.asarray(d1['environment.time'])

        m2, _, _ = myokit.load("shannon_wang_puglisi_weber_bers_2004_a_p.mmt")
        m2.var('reversal_potentials.E_Na_SL').set_rhs(-15.0)
        p2 = myokit.Protocol()
        p2.schedule(level=-1.0, start=self.jump_time2-self.duration, duration=self.duration, period=self.period, multiplier=0)
        
        
        for i, par in enumerate(self.parms2):
            m2.var(par + '.p').set_rhs(parameters[i])
        s2 = myokit.Simulation(m2, p2)
        dt = times1[1] - times1[0]
        d2 = s2.run(times2[-1] - times1[-1] - 10*dt + 0.01, log_times=times2 - times1[-1] - 10*dt)
        v2 = np.asarray(d2['cell.V'])
        t2 = np.asarray(d2['environment.time'])
        
        return d1, d2
    
       
       
    def n_parameters(self):
        return len(self.parms)
        
    def compute_quantities(self, parameters, times):
        split_index = np.argmax(np.diff(times) > 2 * (times[1] - times[0]))
        times1 = times[:split_index + 1]
        times2 = times[split_index + 1:]

        m1, _, _ = myokit.load("shannon_wang_puglisi_weber_bers_2004_a_p.mmt")
        m1.var('reversal_potentials.E_Na_SL').set_rhs(-15.0)
        p1 = myokit.Protocol()
        p1.schedule(level=-1.0, start=self.jump_time1-self.duration, duration=self.duration, period=self.period, multiplier=0)
        
        for i, par in enumerate(self.parms1):
            m1.var(par + '.p').set_rhs(parameters[i])
        
        dy = ['cell.V']
        dp = [param + '.p' for param in self.parms1] 
           
        s1 = myokit.Simulation(m1, p1, sensitivities=(dy, dp))
        
        
        d1, e1 = s1.run(times1[-1] + 0.01, log_times=times1)
        e1 = np.array(e1)
        i_transient = 0
        J = e1[i_transient:,0,:]
        JtJ = np.dot(J.T,J)
        JtJinv = np.linalg.inv(JtJ)
        variance_vec1 = np.diag(JtJinv)

        m2, _, _ = myokit.load("shannon_wang_puglisi_weber_bers_2004_a_p.mmt")
        m2.var('reversal_potentials.E_Na_SL').set_rhs(-15.0)
        p2 = myokit.Protocol()
        p2.schedule(level=-1.0, start=self.jump_time2-self.duration, duration=self.duration, period=self.period, multiplier=0)
        
        
        for i, par in enumerate(self.parms2):
            m2.var(par + '.p').set_rhs(parameters[i])
            
        dy = ['cell.V']
        dp = [param + '.p' for param in self.parms2] 
        
        
        s2 = myokit.Simulation(m2, p2, sensitivities=(dy, dp))
        dt = times1[1] - times1[0]
        
        
        d2, e2 = s2.run(times2[-1] - times1[-1] - 10*dt + 0.01, log_times=times2 - times1[-1] - 10*dt)
        e2 = np.array(e2)
        i_transient = 0
        J = e2[i_transient:,0,:]
        JtJ = np.dot(J.T,J)
        JtJinv = np.linalg.pinv(JtJ)
   #     JtJinv = np.linalg.inv(JtJ)
        variance_vec2 = np.diag(JtJinv)
        print(variance_vec1)
        print(variance_vec2)
        avg_first8 = (variance_vec1 + variance_vec2[:8]) / 2
        print(avg_first8)
        # Take the last element of variance_vec2
        krblock_var = variance_vec2[8]
        CaLblock_var = variance_vec2[9]
        # Combine into final 9-element array
        variance_vec = np.concatenate([avg_first8, [krblock_var], [CaLblock_var]])
        print(variance_vec)
        return variance_vec
        
        
        
        
model = ActionPotentialModel()



def rescale(z,c,d,a,b):
    x=a+(b-a)/(d-c)*(z-c)
    return x


def find_jump_times(t, V, n_points=40, eps=1e-9, t_min=74, t_max=90, min_points=1):
    """
    Find all times t[i] between t_min and t_max such that the next `n_points` values
    are strictly increasing.
    If no segment is found, try decreasing `n_points` down to `min_points`.
    If still nothing found, return an empty list [].
    """
    t = np.asarray(t)
    V = np.asarray(V)
    
    indices = np.where((t >= t_min) & (t <= t_max))[0]
    
    while n_points >= min_points:
        jump_times = []
        for i in indices:
            if i + n_points < len(V):
                if np.all(np.diff(V[i:i+n_points+1]) > eps):
                    jump_times.append(float(t[i]))
        if jump_times:   # if we found any, stop and return
            return jump_times[0]
        n_points -= 1  # relax condition
    
    # If nothing found
    return []




def compute_apd90_single(voltage, time, jump_time):
    """
    Compute APD90 anchored at a known jump_time.
    
    Parameters
    ----------
    voltage : array-like
        Voltage trace
    time : array-like
        Time vector (same length as voltage)
    jump_time : float
        Known upstroke (jump) time
    
    Returns
    -------
    apd90 : float or None
        APD90 duration in same units as `time`
    """
    voltage = np.asarray(voltage)
    time = np.asarray(time)
    
    # ---- Restrict search to AFTER jump_time ----
    start_idx = np.searchsorted(time, jump_time)
    v_seg = voltage[start_idx:]
    t_seg = time[start_idx:]
    if len(v_seg) < 3:
        return None

    # ---- Compute peak and rest in this window ----
    peak_voltage = np.max(v_seg)
    resting_voltage = np.min(voltage)   # use global rest, or np.median before jump_time
    threshold = resting_voltage + 0.1 * (peak_voltage - resting_voltage)

    # ---- Find indices crossing threshold ----
    indices = np.where(v_seg >= threshold)[0]
    if len(indices) == 0:
        return None
    
    v_seg1 = v_seg[indices[0]+1:]
    t_seg1 = t_seg[indices[0]+1:]
    
    indices1 = np.where(v_seg1 <= threshold)[0]
    if len(indices1) == 0:
        return None
    # Upstroke crossing (first index after jump_time)
    t_up = t_seg[indices[0]]
    print(t_up)
    # Downstroke crossing (last index after jump_time)
    t_down = t_seg1[indices1[0]]
    print(t_down)
    
    Vrest = np.mean(v_seg1[indices1[0]:-1])
    apd90 = t_down - t_up
    return apd90, Vrest
    
    
    

def get_experimental_trace(file='Celln_1.csv'):
# Read measured AP trace from file and rescale as appropriate
    try:
        # Attempt to read the CSV file into a pandas DataFrame
        df = pd.read_csv(file)
        # Check if the DataFrame is empty
        if df.empty:
            print(f'{file} is empty.')
            return None,None
        else:
            print('Fitting:   ',f'{file}')
            x = df.iloc[:, 0].values
            y = df.iloc[:, 1].values

            milisec=1000
            x_scaled = milisec*x
            t_front_start, Vrest, Vpeak, Vplateau, jump_times = get_AP_features(x_scaled,y)
            # print(t_front_start) 
            # y_scaled = rescale(y,Vrest, Vpeak,-86.0,20.0)
            # y_scaled = rescale(y,Vrest, Vpeak,-86.0,0.0)
            # y_scaled = rescale(y,Vrest, Vplateau,-86.0,0.0)
            y_scaled = rescale(y,Vrest, Vplateau,-86.0,0.0)
            
            times = x_scaled
            trace = y_scaled
            # plt.figure()
            # plt.xlabel('Time')
            # plt.ylabel('Voltage')
            # plt.plot(times,trace, alpha=0.5)
            # plt.show()     
            
            return times, trace, jump_times
        
    except pd.errors.ParserError:
        # If the file is corrupted and cannot be read, print an error message
        print(f'{file} is corrupted and cannot be read. Moving to next cell.')
        return None,None


def get_AP_features(x,y):
    
    
    
    jump_times = find_jump_times(x, y, n_points=40, t_min=74, t_max=90)
    #print(jump_times) 
    
    
    
    #plt.plot(x, y, label='V',linewidth=.5,marker='o',markersize=1)
    
    # get front time t_front
    dydx = np.gradient(y) / np.gradient(x)
    ## plt.plot(x, dydx, label='dV/dt')
    front_index = np.argmax(dydx[:len(dydx)//2])
    t_front  = x[front_index]
    ## plt.scatter(front_time,dydx[front_index],marker='o', s=30)
    # plt.scatter(t_front,y[front_index],marker='o', s=40)
    
    # get Vpeak
    offset=60
    # plt.scatter(x[front_index+offset],y[front_index+offset],marker='o', s=40)
    Vpeak_index=front_index + np.argmax(y[front_index:front_index+offset])
    Vpeak=y[Vpeak_index]
    # plt.scatter(x[Vpeak_index],Vpeak,marker='o', s=40)
    
    # get t_front_start
    offset=60
    ## plt.scatter(x[front_index-offset],y[front_index-offset],marker='o', s=40)
    front_start_index=(front_index-offset) + np.argmin(y[front_index-offset:front_index])
    t_front_start = x[front_start_index]
    V_front_start = y[front_start_index]
    # plt.scatter(t_front_start,V_front_start, marker='o', s=40)            
    
    # get V_rest=V_prefront
    V_prefront = np.mean(y[:front_start_index])
    # plt.plot(x[:front_start_index],V_prefront * np.ones_like(x[:front_start_index]))
    #Vrest = np.mean(y[-1000:])
    # plt.plot(x[:front_start_index],Vrest * np.ones_like(x[:front_start_index]))
    
    
    apd90, Vrest = compute_apd90_single(y, x, jump_times)
    print(apd90)
    #Vplt = 
    #f_interp = interp1d(x, y, bounds_error=False, fill_value="extrapolate")

    # Compute target time(s)
    #t_target = np.array(jump_times) + apd90 / 2

    # Get corresponding voltages
    #Vplt = f_interp(t_target)
    #plt.plot(x[500:3000], Vplt * np.ones_like(x[500:3000]))
    #print(Vplt)
    
    
    
    t_start = np.array(jump_times)
    #print(t_start)
    t_end = t_start + apd90 / 2.0
    #print(t_end)
    mask = np.where((np.array(x) >= t_start) & (np.array(x) <= t_end))[0]
    #print(mask)
    y = np.asarray(y)
    Vplateau = np.mean(y[mask])
    #plt.axvline(t_start, color="g", linestyle="--", label="t_start")
    #plt.axvline(t_end, color="r", linestyle="--", label="t_end")
    #print(Vplateau)
    #Vplateau = np.mean(y[1000:2000])
    #print(Vplateau)
    #plt.plot(x[500:3000], Vplt * np.ones_like(x[500:3000]))

    ## plt.title(file)
    ## plt.savefig(file + '.png')
    # plt.show()
    # plt.clf()

    return t_front_start, Vrest, Vpeak, Vplateau, jump_times
   



#def return_SSE(cell_file='Celln_1.csv'):
def return_SSE(times,trace,found_parameters,jump_time1,jump_time2):
    # Get measured AP trace from file and rescale as appropriate
#    times,trace=get_experimental_trace(file=cell_file)
    if times is None:
        return 
    else:
        model_local = ActionPotentialModel()
        model_local.jump_time1, model_local.jump_time2 = jump_time1, jump_time2
        # Create an object with links to the model and time series
        problem = pints.SingleOutputProblem(model_local, times, trace)

        SSE = pints.SumOfSquaresError(problem, weights = [1.0/found_parameters[-1]**2.0])
        print("point estimates      = ", found_parameters)
        print(SSE.n_parameters(),found_parameters[0:SSE.n_parameters()])
        print(found_parameters[-1])

        chi2=SSE(found_parameters[0:SSE.n_parameters()])
        print(np.size(trace), np.size(found_parameters[0:SSE.n_parameters()]))
        pvalue = 1-ss.chi2.cdf(chi2, np.size(trace)-np.size(found_parameters))
        print("Goodness of fit chi2, pvalue = ", chi2 , pvalue)

        return pvalue,chi2 





   









    
    
    
def combine_two_traces(file1, file2):
    t1, y1, jump_times1 = get_experimental_trace(file1)
    t2, y2, jump_times2 = get_experimental_trace(file2)
    offset = t1[-1] + 10* (t1[1] - t1[0])
    t2_shifted = t2 + offset
    return np.concatenate([t1, t2_shifted]), np.concatenate([y1, y2])



def split_two_traces(file1, file2):
    t1, y1, jump_times1 = get_experimental_trace(file1)
    t2, y2, jump_times2 = get_experimental_trace(file2)
    
    return y1, y2, jump_times1, jump_times2


def optimise_model_likelihood(times,trace):
    problem = pints.SingleOutputProblem(model, times, trace)



    log_likelihood = pints.GaussianLogLikelihood(problem)
    print(log_likelihood.n_parameters())
    lower = 0.0001*np.ones(log_likelihood.n_parameters())
    upper = 100.0*np.ones(log_likelihood.n_parameters())

    # Set last param (IKrblock) to upper bound = 1.0
    lower[-2] = 0.0001
    upper[-2] = 1.0
    
    lower[-3] = 0.0001
    upper[-3] = 1.0

    boundaries = pints.RectangularBoundaries(lower, upper)
    
    
    
    

    # Select a starting point - typically use the Shannon published values 
    x0=[1,1,1,1,1,1,1,1,0.5,0.9,1]
    
    # x0 = np.random.uniform(0.1, 2.0, log_likelihood.n_parameters())
            
    transformation=pints.LogTransformation(n_parameters=log_likelihood.n_parameters())
            
    # Setup optimization using CMAES (see docs linked above).
    optimiser = pints.OptimisationController(log_likelihood, x0, boundaries=boundaries, method=pints.XNES,transformation=transformation)

    optimiser.set_parallel(parallel=True)
    opti=optimiser.optimiser()
    opti.set_population_size(population_size=25)
    optimiser.set_max_unchanged_iterations(iterations=100, threshold=5e-5)
    found_parameters, found_score = optimiser.run()     
    print('Final score:', found_score)       
    print("parms_to estimate          = ", parms_to_fit + ['sigma_V'])
    print("point estimates, sigma_V   = ", found_parameters)     
        
        

    return found_parameters





def optimise_model_Rmse(times,trace):
    problem = pints.SingleOutputProblem(model, times, trace)
    score = pints.RootMeanSquaredError(problem)



    
    print(model.n_parameters())
    lower = 0.0001*np.ones(model.n_parameters())
    upper = 100.0*np.ones(model.n_parameters())

    # Set last param (IKrblock) to upper bound = 1.0
    lower[-1] = 0.0001
    upper[-1] = 1.0

    boundaries = pints.RectangularBoundaries(lower, upper)
    
    
    
    

    # Select a starting point - typically use the Shannon published values 
    x0=[1,1,1,1,1,1,1,1,0.5]
    
    # x0 = np.random.uniform(0.1, 2.0, log_likelihood.n_parameters())
            

            
    # Setup optimization using CMAES (see docs linked above).
    transformation=pints.LogTransformation(n_parameters=model.n_parameters())
            
    # Setup optimization using CMAES (see docs linked above).
    optimiser = pints.OptimisationController(score, x0, boundaries=boundaries, method=pints.CMAES,  transformation=transformation)
    # optimiser = pints.OptimisationController(score, x0, boundaries=boundaries, method=getattr(pints, opt_method))
    optimiser.set_parallel(parallel=True)
    opti=optimiser.optimiser()
    opti.set_population_size(population_size=65)
    optimiser.set_max_unchanged_iterations(iterations=100, threshold=5e-5)
    
    found_parameters, found_score = optimiser.run()     
    print('Final score:', found_score)       
    print("parms_to estimate          = ", parms_to_fit + ['sigma_V'])
    print("point estimates, sigma_V   = ", found_parameters)     
        
        

    return found_parameters
    
    
    
    
    

def extract_id_and_run(filename):
    match = re.search(r'(\d{6})-LVcells-dof_p_.*?_run(\d+)cell(\d+)', filename)
    if not match:
        return None, None
    trace_key = f"{match.group(1)}_cell{match.group(3)}"
    run_number = int(match.group(2))
    return trace_key, run_number

def find_trace_pairs(root_path):
    all_files = glob.glob(os.path.join(root_path, '**/*.csa'), recursive=True)
    files_by_key = {}
    for f in all_files:
        fname = os.path.basename(f)
        key, run = extract_id_and_run(fname)
        if key is None:
            continue
        files_by_key.setdefault(key, {})[run] = f
    pairs = []
    for key, runs in files_by_key.items():
        for r in sorted(runs):
            if r + 1 in runs:
                pairs.append((runs[r], runs[r + 1]))
    return pairs

if __name__ == '__main__':
    args = docopt(__doc__)
    if args['--fit_data']:
        data_path = args['--data_path']
        output_file = args['--output_file']
        use_rmse = args['--rmse']
        use_likelihood = args['--likelihood']

        pairs = find_trace_pairs(data_path)
        print(f'Found {len(pairs)} trace pairs.')

        # Create output file and write header first
        with open(output_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Cell'] + parms_to_fit)

        for before_file, after_file in pairs:
            print(f'\nProcessing:\n  BEFORE: {before_file}\n  AFTER:  {after_file}')
            try:
                times, trace = combine_two_traces(before_file, after_file)
                
                model.jump_time1, model.jump_time2 = jump_times(before_file, after_file)
                
                if use_likelihood:
                    params = optimise_model_likelihood(times, trace)
                    
            
                
                else:
                    params = optimise_model_Rmse(times, trace)
                cell_name = os.path.basename(after_file)

                # Append result after each fitting
                with open(output_file, 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([cell_name] + list(params))

            except Exception as e:
                print(f'Failed: {e}')

        print(f"\nResults saved to {output_file}")