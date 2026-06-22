# -*- coding: utf-8 -*-
"""
Created on Sat May 30 2026

@author: Mathies Elfrink, adapted from 'Applying HMP - response phase' 

Applying HMP on the decision phase
"""

#import all necessary packages
import os


import matplotlib.pyplot as plt
import mne
from mne.io import read_info

import xarray as xr

import hmp

import numpy as np

import pickle

from pathlib import Path


# %%
"""
preprocessing 1: loading data, splitsing player 1 from player 2
"""


# setting working directory 
epoch_data_path = 'C:/Scriptie_Data/rps' # current wd

# reading .fif filenames in the directory
all_files = [f for f in os.listdir(epoch_data_path) if (f.endswith('.fif') and (f.startswith('decision_phase_')))] # extra test: must be decision_phase epoched

# remove bad participants
all_files.remove('decision_phase_sub-01_player-2_epo.fif')

# filepaths: seperating players - needed to read correct metadata column
p1_files = sorted([epoch_data_path + "/" + f for f in all_files if 'player-1' in f])
p2_files = sorted([epoch_data_path + "/" + f for f in all_files if 'player-2' in f])

# filenames: same as above
p1_names = [os.path.splitext(f)[0] for f in all_files if 'player-1' in f]
p2_names = [os.path.splitext(f)[0] for f in all_files if 'player-2' in f]

# sampling frequency
sfreq = 200

# load player 1 data
epoch_data_p1 = hmp.io.read_mne_data(
    p1_files, data_format='epochs', sfreq=sfreq,
    verbose=True, subj_name=p1_names
)

# load player 2 data
epoch_data_p2 = hmp.io.read_mne_data(
    p2_files, data_format='epochs', sfreq=sfreq,
    verbose=True, subj_name=p2_names
)


#%%
"""
Preprocessing 2: adding rt=2 column, adding outcome from participant perspective
"""


# After loading but before concatenating:
#   Add a common 'rt' coordinate to each dataset pointing to the right column 
"""swap p1 and p2, as this is swapped in the original data"""

#   Add an 'outcome_perspective' column, to get the outcome from every participants perspective the same.
"""outcome possibilities"""
# "1": "draw"
# "2": "player 1 wins"
# "3": "player 2 wins"

#======= PLAYER 1 ========
epoch_data_p1 = epoch_data_p1.assign_coords(
    rt=xr.full_like(epoch_data_p1['player2_rt'], fill_value=2.0),
    outcome_perspective=epoch_data_p1['outcome']  # 1=draw, 2=win, 3=loss
)

#======= PLAYER 2 ========
outcome_p2 = epoch_data_p2['outcome'].values.copy()
remapped_p2 = outcome_p2.copy()
remapped_p2[outcome_p2 == 2] = 3  # a win for p1, is a loss for p2
remapped_p2[outcome_p2 == 3] = 2  # a loss for p1, is a win for p2

epoch_data_p2 = epoch_data_p2.assign_coords(
    rt=xr.full_like(epoch_data_p2['player1_rt'], fill_value=2.0),
    outcome_perspective=(["participant", "epoch"], remapped_p2),  # now: 1=draw, 2=win, 3=loss
)


#%%
"""
TEST for preprocessing: no missing data?
"""

## CHECK FOR MISSING TRIALS------------------
# Check per-participant trial counts after loading, before concat
print("P1 participants:", epoch_data_p1.participant.values)
print("P1 n_trials attr:", epoch_data_p1.attrs['n_trials'])

print("P2 participants:", epoch_data_p2.participant.values)  
print("P2 n_trials attr:", epoch_data_p2.attrs['n_trials'])

# Also check for NaN epochs in the RT coordinates
print("NaNs in p1 rt:", np.isnan(epoch_data_p1.data.values).sum())
print("NaNs in p2 rt:", np.isnan(epoch_data_p2['player2_rt'].values).sum())
##----------------------------------------


#%%
"""
preprocessing 3: shifting outcome to a previous_outcome, concatenating on win/loss/draw
"""
# Add new coordinate: outcome from previous trial

# === PLAYER 1 ===
prev_p1 = np.roll(epoch_data_p1['outcome_perspective'].values, shift=1, axis=1) # Shift for each participant (shift the epoch axis so, axis=1)
prev_p1[:, 0] = -1 # first trial does not have a prev trial

epoch_data_p1 = epoch_data_p1.assign_coords(
    prev_outcome=(["participant", "epoch"], prev_p1)
)

# === PLAYER 2 ===
prev_p2 = np.roll(epoch_data_p2['outcome_perspective'].values, shift=1, axis=1) # see PLAYER 1
prev_p2[:, 0] = -1 # see PLAYER 1

epoch_data_p2 = epoch_data_p2.assign_coords(
    prev_outcome=(["participant", "epoch"], prev_p2)
)

# Concatenate along the participant dimension
epoch_data = xr.concat([epoch_data_p1, epoch_data_p2], dim='participant')


# === THREE SEPERATE CONCATENATIONS ===
# 1 = draw, 2 = win, 3 = loss (from participant-perspective)

epoch_data_win = xr.concat(
    [epoch_data_p1.where(epoch_data_p1['prev_outcome'] == 2, drop=True), # note: drop=True throws away non-selected trials in stead of setting them at NaN
     epoch_data_p2.where(epoch_data_p2['prev_outcome'] == 2, drop=True)],
    dim='participant'
)

epoch_data_loss = xr.concat(
    [epoch_data_p1.where(epoch_data_p1['prev_outcome'] == 3, drop=True),
     epoch_data_p2.where(epoch_data_p2['prev_outcome'] == 3, drop=True)],
    dim='participant'
)

epoch_data_draw = xr.concat(
    [epoch_data_p1.where(epoch_data_p1['prev_outcome'] == 1, drop=True),
     epoch_data_p2.where(epoch_data_p2['prev_outcome'] == 1, drop=True)],
    dim='participant'
)

#%%
"""
preprocessing 4: concat all for the collapsed model
"""
epoch_data_collapsed = xr.concat([epoch_data_win, epoch_data_loss, epoch_data_draw], dim='epoch')

# Reset epoch index to be unique - necessary for HMP
epoch_data_collapsed = epoch_data_collapsed.assign_coords(
    epoch=np.arange(epoch_data_collapsed.sizes['epoch'])
)


# %%
"""
reset models
"""
results = {}
models = {}
#%%

"""
applying HMP: fitting the model for the three conditions
"""


for label, data in [('win_1000_EventModel_n=3_sp=8', epoch_data_win), ('loss_1000_EventModel_n=3_sp=8', epoch_data_loss), ('draw_1000_EventModel_n=3_sp=8', epoch_data_draw)]:
    # Step 1: Preprocess using PCA
    # >> set ofset_end to change RT
    preprocessed = hmp.transformers.ProjPCA(data, n_comp=32, interval_id='rt', offset_end=-1.0, reject_threshold=1e-4, min_duration=0.2, max_duration=2.0) 

    # Step 2: Create expected pattern
    pattern = hmp.patterns.HalfSine.create_expected(sfreq=epoch_data_win.sfreq, width=50, location=50)
        
    # Step 3: Build TrialData
    trial_data = hmp.trialdata.TrialData.from_transformer(preprocessed, pattern=pattern.template)

    # Step 4: Fit model 
    #   a) EventModel - set number of events 
    #   b) CumulativeMethod - let HMP figure out this number (here: resulted in duplicates)
    n_events = 3
    model = hmp.models.EventModel(pattern, n_events=n_events, starting_points=8, max_scale=41) # max_scale = (n_events + 1) * 10 + 1
    #model = hmp.models.CumulativeMethod(pattern) 
    _, estimates = model.fit_transform(trial_data)
    
    results[label] = (estimates) # estimates for showing end result
    #models[label] = model # model for showing log-likelihood plot


# %%

"""
applying HMP: fitting the model for the collapsed dataset
"""

for label, data in [('collapsed_1000_EventModel_n=3_sp=8', epoch_data_collapsed)]:
    # Step 1: Preprocess using PCA
    # >> set ofset_end to change RT
    preprocessed = hmp.transformers.ProjPCA(data, n_comp=32, interval_id='rt', offset_end=-1.0, reject_threshold=1e-4, min_duration=0.2, max_duration=2.0) 

    # Step 2: Create expected pattern
    pattern = hmp.patterns.HalfSine.create_expected(sfreq=epoch_data_collapsed.sfreq, width=50, location=50)
        
    # Step 3: Build TrialData
    trial_data = hmp.trialdata.TrialData.from_transformer(preprocessed, pattern=pattern.template)

    # Step 4: Fit model 
    #   a) EventModel - set number of events 
    #   b) CumulativeMethod - let HMP figure out this number (here: resulted in duplicates)
    n_events = 3
    model = hmp.models.EventModel(pattern, n_events=n_events, starting_points=8, max_scale=41) # max_scale = (n_events + 1) * 10 + 1
    #model = hmp.models.CumulativeMethod(pattern)
    _, estimates = model.fit_transform(trial_data)
    
    results[label] = (estimates) # estimates for showing end result
    models[label] = model # model for showing log-likelihood plot
    
    

#%%
"""
Analyse data - getting average event timings -> conditions (similar for collapsed)
"""

timings = {}

for label in ['win', 'loss', 'draw']:
    timing = hmp.utils.event_times(results[label], mean=True, as_time=True)
    rounded_timing = np.round(timing.values / 5) * 5 # round similar to the cumulative method
    timings[label] = rounded_timing


#%%

"""
visualize data - topographies -> conditions
"""

# set working dir (not sure why this is necessary, maybe for by read_info())
os.chdir('C:/Scriptie_Data/rps') 

info = read_info(all_files[0], verbose=False)

all_epoch_data = {
    'win':  epoch_data_win,
    'loss': epoch_data_loss,
    'draw': epoch_data_draw,
}

suffixes = list(dict.fromkeys(k.split('_', 1)[1] for k in results))

for suffix in suffixes:
    for outcome in ['win', 'loss', 'draw']:
        key = f"{outcome}_{suffix}"
        if key in results:
            hmp.visu.plot_topo_timecourse(
                all_epoch_data[outcome],
                results[key],
                info,
                as_time=True,
                event_lines=False,
                title=f"Topographical Timecourse: {outcome.capitalize()} {suffix}",
                max_time=1000, # fix topography-axis
                vmin = -1e-5, # fix max and min EEG signal
                vmax = 1e-5
            )

#%%
"""
visualize data - topographies -> collapsed
"""
# set working dir (not sure why this is necessary, maybe for by read_info())
os.chdir('C:/Scriptie_Data/rps') 

info = read_info(all_files[0], verbose=False)
hmp.visu.plot_topo_timecourse(epoch_data_collapsed, 
                              results['collapsed_1000_EventModel_n=3_sp=8'], 
                              info, 
                              as_time=True, 
                              event_lines=False, 
                              title="Topographical Timecourse: Collapsed",
                              max_time=1000,
                              vmin = -1e-5, 
                              vmax = 1e-5)



#%%
"""
visualize data - timeline average probability for each event in ms -> conditions
"""


# set working dir (not sure why this is necessary, maybe for by read_info())
os.chdir('C:/Scriptie_Data/rps') 

info = read_info(all_files[0], verbose=False)

all_epoch_data = {
    'win':  epoch_data_win,
    'loss': epoch_data_loss,
    'draw': epoch_data_draw,
}

suffixes = list(dict.fromkeys(k.split('_', 1)[1] for k in results))

for suffix in suffixes:
    for outcome in ['win', 'loss', 'draw']:
        key = f"{outcome}_{suffix}"
        if key in results:
            for event in range(3):
                estimates = results[key]
                data = estimates.sel(event=event).mean('trial')
                time_ms = data.sample.values / sfreq * 1000  # samples -> ms
                plt.plot(time_ms, data.values, label=f"Event {event+1}")

            plt.ylabel('P(event)')
            plt.ylim(top=0.025)
            plt.xlabel('Time (ms)')
            plt.title(f'Average Probability Distributions: {outcome.capitalize()}')
            plt.legend()
            plt.show()



#%%
"""
log-likelihood plot 
"""
# setting the plot
fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=False)

# setting the colors... because why not pick some nice colors :)
colors = {'win': 'steelblue', 'loss': 'tomato', 'draw': 'goldenrod'}


"""
#TEST

models={}
models['win'] = x
"""

# loop over all models, plot the loglikelihood
for ax, label in zip(axes, ['win', 'loss', 'draw']):
    model = models[label]

    # Extract n_events and log-likelihood from each submodel
    cumulative_res = np.array([
        [submodel.n_events, submodel.lkhs[0]]
        for submodel in model.submodels
    ])

    n_events_vals = cumulative_res[:, 0]
    ll_vals = cumulative_res[:, 1]

    ax.plot(n_events_vals, ll_vals, 'o-', color=colors[label], linewidth=2, markersize=6)

    # Mark the final number of events for clarity 
    final_n = model.submodels[-1].n_events
    final_ll = model.submodels[-1].lkhs[0]
    ax.plot(final_n, final_ll, 'o', color=colors[label], markersize=12,
            markeredgecolor='black', markeredgewidth=1.5, label=f'Number of final events: {int(final_n)} events')

    ax.set_xlabel('Number of Events', fontsize=12)
    ax.set_ylabel('Log-likelihood', fontsize=12)
    ax.set_title(f'Condition: {label}', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

plt.suptitle('Log-likelihood per conditie (CumulativeMethod)', fontsize=14, y=1.0)
plt.tight_layout()
plt.savefig('loglikelihood_per_conditie.png', dpi=150, bbox_inches='tight')
plt.show()


#%%
"""
save single estimates as pickle
"""
with open("collapsed_1000_EventModel_n=6_sp=8.pkl", 'wb') as output:
    pickle.dump(estimates, output)
    
#%%
"""
open single estimates pickle
"""
epoch_data_path = 'C:/Scriptie_Data/rps' # current wd

with open("C:/Scriptie_Data/rps/collapsed_1000_EventModel_n=3_sp=8.pkl", 'rb') as output:
    results['collapsed_1000_EventModel_n=3_sp=8'] = pickle.load(output)
    
# %%

"""
save models as pickle files
"""
os.chdir('C:/Scriptie_Data/rps')

for label, model in models.items():
    with open(f"model_{label}.pkl", 'wb') as output:
        pickle.dump(model, output)
#%%
"""
loading pickled models and estimates
"""

rps_path = Path('C:/Scriptie_Data/rps')

models = {}
for filepath in rps_path.iterdir():
    if filepath.stem.startswith('model_'):
        label = filepath.stem.removeprefix('model_')
        with open(filepath, 'rb') as file:
            models[label] = pickle.load(file)
    
results = {}
for filepath in rps_path.iterdir():
    if filepath.stem.startswith('estimates_final_'):
        label = filepath.stem.removeprefix('estimates_final_')
        with open(filepath, 'rb') as file:
            results[label] = pickle.load(file)
    
