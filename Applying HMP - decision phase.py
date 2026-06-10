# -*- coding: utf-8 -*-
"""
Created on Sat May 30 2026

@author: Mathies Elfrink, adapted from 'Applying HMP - final'

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


# %%

"""
fitting the model for all three datasets - 
"""
results = {}
models = {}

for label, data in [('win', epoch_data_win), ('loss', epoch_data_loss), ('draw', epoch_data_draw)]:
    # Step 1: Preprocess (set n_comp manually, more is more detailed, therefore also more likely to introduce noise)
    # preprocessed = hmp.preprocessing.Standard(epoch_data, n_comp=32, apply_standard=True)
    preprocessed = hmp.transformers.ProjPCA(data, n_comp=32, interval_id='rt', offset_end=0, reject_threshold=1e-4, min_duration=0.2, max_duration=2.0) # EDIT was: offset_end = 0.05 or 0.1, 'worked' with =0, -0.1 did not give better results

    # Step 2: Create expected pattern
    pattern = hmp.patterns.HalfSine.create_expected(sfreq=epoch_data.sfreq, width=50, location=50)
        
    # Step 3: Build TrialData
    trial_data = hmp.trialdata.TrialData.from_transformer(preprocessed, pattern=pattern.template)

    # Step 4: Fit model 
    #   a) EventModel - set number of events 
    #   b) CumulativeMethod - let HMP figure out this number (here: resulted in duplicates)
    n_events = 4
    model = hmp.models.EventModel(pattern, n_events=n_events, starting_points=1, max_scale=10) #edit: starting_points was 10, max_scale was 51. Worked with startingpoints=1, max_scale=10. With starting_points=3, max_scale=20 too much. 
    #model = hmp.models.CumulativeMethod(pattern, max_n_events=4) # EDIT: max_n_events toegevoegd
    _, estimates = model.fit_transform(trial_data)
    
    results[label] = (estimates) # estimates for showing end result
    models[label] = model # model for showing log-likelihood plot


#%%
"""
Analyse data - getting average event timings
"""

timings = {}

for label in ['win', 'loss', 'draw']:
    timing = hmp.utils.event_times(results[label], mean=True, as_time=True)
    rounded_timing = np.round(timing.values / 5) * 5 # round similar to the cumulative method
    timings[label] = rounded_timing

#%%
"""
visualize data - topographies
"""
# set working dir (not sure why this is necessary, maybe for by read_info())
os.chdir('C:/Scriptie_Data/rps') 

info = read_info(all_files[0], verbose=False)
hmp.visu.plot_topo_timecourse(epoch_data_win, results['win'], info, as_time=True, event_lines=False, title="A. Topographical Timecourse: Win")
hmp.visu.plot_topo_timecourse(epoch_data_loss, results['loss'], info, as_time=True, event_lines=False, title="B. Topographical Timecourse: Loss")
hmp.visu.plot_topo_timecourse(epoch_data_draw, results['draw'], info, as_time=True, event_lines=False, title="C. Topographical Timecourse: Draw")


#%%
"""
visualize data - timeline average probability for each event at each time sample
"""

for event in range(7):
    plt.plot(estimates.sel( event=event).mean('trial'), label=f"Event{event+1}")
plt.ylabel('P(event)')
plt.xlabel('Time (sample)')
plt.legend()
plt.show()

#%%
"""
visualize data - timeline probability for each event for one trial
"""

for event in range(7):
    plt.plot(estimates.sel(trial=('decision_phase_sub-01_player-1_epo',0), event=event), label=f"Event {event+1}") # TEST: 'TEST_'
plt.ylabel('P(event)')
plt.xlabel('Time (sample)')
plt.xlim(0,400)
plt.legend()
plt.show()



#%%
x = model

#%%
"""
log-likelihood plot 
"""
# setting the plot
fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=False)

# setting the colors
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

    # Mark the last number of events for clarity 
    final_n = model.submodels[-1].n_events
    final_ll = model.submodels[-1].lkhs[0]
    ax.plot(final_n, final_ll, 'o', color=colors[label], markersize=12,
            markeredgecolor='black', markeredgewidth=1.5, label=f'Number of final events: {int(final_n)} events')

    ax.set_xlabel('Number of Events', fontsize=12)
    ax.set_ylabel('Log-likelihood', fontsize=12)
    ax.set_title(f'Condition: {label}', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

plt.suptitle('Log-likelihood per conditie (CumulativeMethod)', fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig('loglikelihood_per_conditie.png', dpi=150, bbox_inches='tight')
plt.show()
# %%

"""
save estimates as pickle files
"""

for label, estimates in results.items():
    with open(f"estimates_{label}.pkl", 'wb') as output:
        pickle.dump(estimates, output)
        
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
loading pickled model and estimates
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
    if filepath.stem.startswith('estimates_'):
        label = filepath.stem.removeprefix('estimates_')
        with open(filepath, 'rb') as file:
            results[label] = pickle.load(file)
    
#%%

"""
loading data for 1 participant
"""


# setting working directory 
epoch_data_path = 'C:/Scriptie_Data/rps' # TEST: '/gedai_processed_data'

# reading .fif filenames in the directory

files = [f for f in os.listdir(epoch_data_path) if f.endswith('.fif')]
s1_p1_file = epoch_data_path + "/" + files[0]

p1_name = os.path.splitext(s1_p1_file)[0]


# sampling frequency
sfreq = 200

# load player 1 data
epoch_data_p1 = hmp.io.read_mne_data(
    s1_p1_file, data_format='epochs', sfreq=sfreq,
    verbose=True, subj_name=p1_name
)


# Add a common 'rt' coordinate to each dataset pointing to the right column
"""use p2 rt as this is swapped in the eeg data"""

epoch_data_p1 = epoch_data_p1.assign_coords(
    rt=epoch_data_p1['player2_rt']
)



## CHECK FOR MISSING TRIALS------------------
# Check per-participant trial counts after loading, before concat
print("P1 participants:", epoch_data_p1.participant.values)
print("P1 n_trials attr:", epoch_data_p1.attrs['n_trials'])

# Also check for NaN epochs in the RT coordinates
print("NaNs in p1 rt:", np.isnan(epoch_data_p1.data.values).sum())
##----------------------------------------



# %%

"""
fitting the model pipeline for participant 1
"""

# Step 1: Preprocess (set n_comp manually, more is more detailed, therefore also more likely to introduce noise)
# preprocessed = hmp.preprocessing.Standard(epoch_data, n_comp=32, apply_standard=True)
preprocessed = hmp.transformers.ProjPCA(epoch_data_p1, n_comp=32, interval_id='rt', offset_end=0.1, reject_threshold=1e-4, min_duration=0.2, max_duration=2.0)

# Step 2: Create expected pattern
pattern = hmp.patterns.HalfSine.create_expected(sfreq=epoch_data_p1.sfreq, width=50, location=50)
    
# Step 3: Build TrialData
trial_data = hmp.trialdata.TrialData.from_transformer(preprocessed, pattern=pattern.template)

# Step 4: Fit model (start with n_events=4, or loop over values for model comparison)
n_events = 4
#model = hmp.models.EventModel(pattern, n_events=n_events, starting_points=10, max_scale=51)
model = hmp.models.CumulativeMethod(pattern)
fitted_model, estimates = model.fit_transform(trial_data)

#%%
"""
visualize data - topographies for participant 1
"""
# set working dir (not sure why this is necessary, maybe caused by read_info())
os.chdir('C:/Scriptie_Data/rps') # TEST: '/gedai_processed_data'

info = read_info(all_files[0], verbose=False)
hmp.visu.plot_topo_timecourse(epoch_data_p1, estimates, info, as_time=True)


#%%
"""
visualize data - timeline average probability for each event at each time sample for participant 1
"""

for event in range(5):
    plt.plot(estimates.sel( event=event).mean('trial'), label=f"Event{event+1}")
plt.ylabel('P(event)')
plt.xlabel('Time (sample)')
plt.legend()
plt.show()