
#!/bin/env python

"""
Plot model fits vs experimental traces for parameter estimates.

Reads a parameter CSV (comma-delimited, header row),
finds the corresponding BEFORE and AFTER experimental traces,
combines them, simulates the model with the estimated parameters,
and plots experiment vs model in 3x3 grids per PDF page.
"""

import os
import re
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

# Import from your fitting script
from updated_a01_fit_models_to_data_auto import ActionPotentialModel, get_experimental_trace, combine_two_traces, parms_to_fit, return_SSE, split_two_traces, jump_times

# ===== User settings =====
results_csv = "/home/pgrad1/2712549y/Parameter_Estimation/test_paramter_estimate_drug_300nM/EstimatedParVals_210526_run14_15_27_CaLblock.csv"
data_folder = "/home/pgrad1/2712549y/Parameter_Estimation/test_paramter_estimate_drug_300nM/data_300nM_210526_run14_15/"
output_prefix = "CaL_test_data_300nM_210526_run14_15"  # final PDF: model_vs_experiment_fits_page.pdf

# Initialize model
model = ActionPotentialModel()

# ===== Helper: parse cell ID =====
def parse_cell_id(cell_id):
    m = re.search(r"(\d{6}).*?_run(\d+)cell(\d+)", cell_id)
    if not m:
        # Another regex for a different ID pattern
        m = re.search(r"(\d{6}).*?_run(\d+)cell(\d+).csa", cell_id)
        if not m:
            raise ValueError(f"Could not parse cell id: {cell_id}")
    date_str, run_after, cell_num = m.group(1), int(m.group(2)), int(m.group(3))
    return date_str, run_after, cell_num

# ===== Load results CSV =====
# Define the column names based on your CSV structure
col_names = ['cell_id', 'INaCa','INaK','IClb','ICaL','Itos','IK1','IKs','IKr','IKrb','ICaLb','sigma']
results_df = pd.read_csv(results_csv, sep=",", header=None, names=col_names)

# Parameter columns
param_cols = ['INaCa','INaK','IClb','ICaL','Itos','IK1','IKs','IKr','IKrb','ICaLb']


param_updated = [
    r'\hat{\alpha}_{NaCa}', r'\hat{\alpha}_{NaK}', r'\hat{\alpha}_{Clb}', 
    r'\hat{\alpha}_{CaL}', r'\hat{\alpha}_{tos}', r'\hat{\alpha}_{K1}', 
    r'\hat{\alpha}_{Ks}', r'\hat{\alpha}_{Kr}', r'\phi_{Kr}', r'\phi_{CaL}'
]

parameters_index = ['INaCa','INaK','IClb','ICaL','Itos','IK1','IKs','IKr','IKrb','ICaLb','sigma']












# ===== Plotting setup =====
ncols, nrows = 3, 3
plot_idx = 0
fig, axes = plt.subplots(nrows, ncols, figsize=(12, 9))
axes = axes.flatten()

output_file = f"{output_prefix}.pdf"
pdf = PdfPages(output_file)

# ===== Loop over each cell =====
for idx, row in results_df.iterrows():
    try:
        params  = row[param_cols].values.astype(float)
        
        parameters_all = row[parameters_index].values.astype(float)
        
        print(parameters_all)
        cell_id = str(row['cell_id'])
        sigma   = float(row['sigma'])
    except Exception as e:
        print(f"Skipping row {idx} due to parsing error: {e}")
        continue

    # Parse cell ID
    try:
        date_str, run_after, cell_num = parse_cell_id(cell_id)
        print(date_str)
        print(run_after)
        print(cell_num)
    except ValueError as e:
        print(e)
        continue

    run_before = run_after - 1

    # Filename patterns
    before_pattern = f".*?{date_str}.*?_run{run_before}cell{cell_num}.csa"
    after_pattern  = f".*?{date_str}.*?_run{run_after}cell{cell_num}.csa"

    before_file, after_file = None, None
    for fname in os.listdir(data_folder):
        if re.search(before_pattern, fname):
            before_file = os.path.join(data_folder, fname)
        elif re.search(after_pattern, fname):
            after_file = os.path.join(data_folder, fname)

    if not before_file or not after_file:
        print(f"Missing before/after files for {cell_id}")
        continue

    # Load experimental trace
    times_exp, trace_exp = combine_two_traces(before_file, after_file)
    #plt.plot(times_exp, trace_exp)
    #plt.show()
    # calculate p value
    jump_time1, jump_time2 = jump_times(before_file, after_file)
    model.jump_time1, model.jump_time2 = jump_times(before_file, after_file)
    #model.jump_time1, model.jump_time2 = 79.69999999999999,79.5
    pvalue,chi2= return_SSE(times_exp,trace_exp, parameters_all, jump_time1, jump_time2)
    
    print(pvalue)
    
    # calculate stderr
    variance_vec = model.compute_quantities(params, times_exp)
    
    stderr= sigma*np.sqrt(variance_vec)
    
    score = sigma
    
    # split traces
    t_b,v_b,t_d,v_d  = model.simulate_quantity(params, times_exp)
    v_b_exp, v_d_exp, _, _ = split_two_traces(before_file, after_file)
    
    _, baseline_b, _, baseline_d = model.simulate_quantity(np.ones_like(params), times_exp)
    
    if times_exp is None or trace_exp is None:
        print(f"Skipping {cell_id} due to read error")
        continue

    # Simulate model
    try:
        trace_sim = model.simulate(params, times_exp)
        
        baseline = model.simulate(np.ones_like(params), times_exp)
        
    except Exception as e:
        print(f"Simulation failed for {cell_id}: {e}")
        continue

    # ===== Plot: split panel into 2 subplots (trace + params) =====
    ax = axes[plot_idx]
    
    # remove placeholder so we can reuse its space cleanly
    ax.remove()
    
    
    gs = ax.get_subplotspec().subgridspec(1, 2, width_ratios=[7, 1], wspace=0.03)
    # Left panel: further split vertically into before after
    gs_trace = gs[0].subgridspec(1, 2, width_ratios=[1,1], wspace=0.06)
    ax_trace_before = fig.add_subplot(gs_trace[0])
    ax_trace_after  = fig.add_subplot(gs_trace[1])
    
    ax_params = fig.add_subplot(gs[1])

    # Left panel: experiment vs model 
    
    ax_trace_before.fill_between(t_b,v_b-score,v_b+score,facecolor='orangered',edgecolor='none',alpha=0.4)
    ax_trace_before.plot(t_b, v_b_exp, 'k', lw=0.8, alpha=0.6, label=r'data ($\hat\sigma=${:.2f})'.format(sigma))
    ax_trace_before.plot(t_b, v_b, 'r', lw=1.5, label=r'fit ($p=${:.2f})'.format(pvalue))
    ax_trace_before.plot(t_b, baseline_b ,color='tab:blue',linewidth=0.7,linestyle='dashed',label='baseline')
    
    ax_trace_before.set_title(f"uid: {date_str} run{run_before} cell{cell_num}", fontsize=6)
    ax_trace_before.set_xlabel("Time (ms)", fontsize=7)
    ax_trace_before.set_ylabel("Voltage (mV)", fontsize=7)
    ax_trace_before.set_ylim([-100, 28])
    ax_trace_before.spines['right'].set_visible(False)
    #ax_trace_before.legend()
    
    
    ax_trace_after.fill_between(t_d,v_d-score,v_d+score,facecolor='orangered',edgecolor='none',alpha=0.4)
    ax_trace_after.plot(t_d, v_d_exp, 'k', lw=0.8, alpha=0.6, label=r'data ($\hat\sigma=${:.2f})'.format(sigma))
    ax_trace_after.plot(t_d, v_d, 'r', lw=1.5, label=r'fit ($p=${:.2f})'.format(pvalue))
    ax_trace_after.plot(t_d, baseline_d ,color='tab:blue',linewidth=0.7,linestyle='dashed',label='baseline')
    
    

    ax_trace_after.set_title(f"uid: {date_str} run{run_after} cell{cell_num}", fontsize=6)
    ax_trace_after.set_xlabel("Time (ms)", fontsize=7)
    ax_trace_after.set_ylim([-100, 28])
    ax_trace_after.set_yticks([])          
    ax_trace_after.set_yticklabels([])    
    ax_trace_after.spines['left'].set_visible(False) 
#    ax_trace_after.set_ylabel("Voltage (mV)", fontsize=7)
    ax_trace_after.legend(fontsize=5, loc='upper right', frameon=False)  



    # --- Right panel: bar chart of parameters ---
    y_pos = np.arange(len(param_cols))
    bars = ax_params.barh(y_pos, params, color='orangered', alpha=0.8)
    
    
    
    # Label with parameter name and value
    labels = [rf"${p} = {val:.2g}$" for p, val in zip(param_updated, params)]
    
    ax_params.errorbar(params, labels, xerr=(stderr,stderr), fmt='none', color='black', capsize=1,capthick=0.5, elinewidth=0.4)
    
    
    ax_params.set_yticks(y_pos)
    ax_params.set_yticklabels(labels, fontsize=5)

    ax_params.set_xscale("log")  # optional, if params span many orders of magnitude
    
    ax_params.set_xlabel("", fontsize=6)
    ax_params.tick_params(axis='y',labelright=True,labelleft=False,left=False,right=True,labelsize=6,direction='in',pad=1)
    ax_params.tick_params(axis='y', which='both', left=False, right=False, pad=1,labelrotation=0,direction='in',labelsize=5)
    ax_params.tick_params(axis='x',which='both',labelright=True,labelleft=False,left=False,right=True,labelsize=6,direction='in',pad=1)
    
    # --- Formatting ---
    ax_params.minorticks_on()
    ax_params.invert_yaxis() # match order
    
    fig.subplots_adjust(wspace=0.03)

    plot_idx += 1

    # Save page when full
    if plot_idx == ncols * nrows:
        plt.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)
        fig, axes = plt.subplots(nrows, ncols, figsize=(12, 9))
        axes = axes.flatten()
        plot_idx = 0

# Save any remaining partial page
if plot_idx > 0:
    for j in range(plot_idx, ncols * nrows):
        axes[j].axis("off")
    plt.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)

pdf.close()
print(f"All pages saved to {output_file}")
