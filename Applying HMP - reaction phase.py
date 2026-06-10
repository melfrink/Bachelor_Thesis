# -*- coding: utf-8 -*-
"""
Created on Sat Apr 25 10:05:16 2026

@author: Mathies Elfrink, adapted from https://hmp.readthedocs.io/en/latest/notebooks/3-Applying_HMP_to_real_data.html

Applying HMP - Reaction phase
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

# %%

"""
sorting data on players, processing it seperately 
"""


# setting working directory 
epoch_data_path = 'C:/Scriptie_Data/rps' # TEST: '/gedai_processed_data'

# reading .fif filenames in the directory
all_files = [f for f in os.listdir(epoch_data_path) if f.endswith('.fif')]

# remove bad participants
all_files.remove('sub-01_player-2_epo.fif')

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


# After loading but before concatenating:
# Add a common 'rt' coordinate to each dataset pointing to the right column
"""EDIT: swap p1 and p2, as this is swapped in the original data"""

epoch_data_p1 = epoch_data_p1.assign_coords(
    rt=epoch_data_p1['player2_rt']
)
epoch_data_p2 = epoch_data_p2.assign_coords(
    rt=epoch_data_p2['player1_rt']
)

"""
EDIT: wisselen RT van player 1 en player 2, omdat deze verkeert andersom in EEG zitten

epoch_data_p1 = epoch_data_p1.assign_coords(
    rt=epoch_data_p1['player2_rt']
)
epoch_data_p2 = epoch_data_p2.assign_coords(
    rt=epoch_data_p2['player1_rt']
)
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



# Concatenate along the participant dimension
epoch_data = xr.concat([epoch_data_p1, epoch_data_p2], dim='participant')


# %%

"""
fitting the model pipeline
"""

# Step 1: Preprocess (set n_comp manually, more is more detailed, therefore also more likely to introduce noise)
# preprocessed = hmp.preprocessing.Standard(epoch_data, n_comp=32, apply_standard=True)
preprocessed = hmp.transformers.ProjPCA(epoch_data, n_comp=32, interval_id='rt', offset_end=0.1, reject_threshold=1e-4, min_duration=0.2, max_duration=2.0) # EDIT was: offset_end = 0.05

# Step 2: Create expected pattern
pattern = hmp.patterns.HalfSine.create_expected(sfreq=epoch_data.sfreq, width=50, location=50)
    
# Step 3: Build TrialData
trial_data = hmp.trialdata.TrialData.from_transformer(preprocessed, pattern=pattern.template)

# Step 4: Fit model (start with n_events=4, or loop over values for model comparison)
# n_events = 4
#model = hmp.models.EventModel(pattern, n_events=n_events, starting_points=10, max_scale=51)
model = hmp.models.CumulativeMethod(pattern)
fitted_model, estimates = model.fit_transform(trial_data)

#%%
"""
visualize data - topographies
"""
# set working dir (not sure why this is necessary, maybe for by read_info())
os.chdir('C:/Scriptie_Data/rps') 

info = read_info(all_files[0], verbose=False)
hmp.visu.plot_topo_timecourse(epoch_data, estimates, info, as_time=True, event_lines=False, title="A. Topographical Timecourse")

#%%
"""
visualize data - timeline average probability for each event in ms
"""

for event in range(3):
    data = estimates.sel(event=event).mean('trial')
    time_ms = data.sample.values / sfreq * 1000  # samples -> ms
    plt.plot(time_ms, data.values, label=f"Event{event+1}")

plt.ylabel('P(event)')
plt.xlabel('Time (ms)')
plt.title('B. Average Probability Distributions')
plt.legend()
plt.show()

#%%
"""
visualize data - timeline probability for each event for one trial
"""

for event in range(3):
    data = estimates.sel(trial=('sub-01_player-1_epo', 0), event=event)
    time_ms = data.sample.values / sfreq * 1000
    plt.plot(time_ms, data.values, label=f"Event {event+1}")

plt.ylabel('P(event)')
plt.xlabel('Time (ms)')
plt.xlim(0, 400 / sfreq * 1000)  # 0 tot 2000 ms
plt.title('C. Single Trial Probability Distributions')
plt.legend()
plt.show()

# %%
"""
save model and estimates as pickle file
"""

with open("model.pkl", 'wb') as output:
    pickle.dump(model, output)


# Same goes fpr the estimates
with open("estimates.pkl", 'wb') as output:
    pickle.dump(estimates, output)

#%%
"""
loading pickled model and estimates
"""

with open('C:/Scriptie_Data/rps/model.pkl', 'rb') as file:
    model = pickle.load(file)
    
with open('C:/Scriptie_Data/rps/estimates.pkl', 'rb') as file:
    estimates = pickle.load(file)



    
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
