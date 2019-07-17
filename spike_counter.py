# -*- coding: utf-8 -*-

"""
This file contains the SpikeCounter Python code.

Authors:
Vanessa D'Amario      <vanessa.damario@dibris.unige.it>
Gianvittorio Luria    <luria@dima.unige.it>
"""


import numpy as np
import os.path as op
import pandas as pd
import datetime
import pickle
import mne
import matplotlib.pyplot as plt
from alphacsc import BatchCDL
from alphacsc.utils.convolution import construct_X_multi


class SpikeCounter(object):

    def __init__(self, datapath, brefpath, filt_l_f=None, filt_h_f=None, t_vis=30,
                 csc_ftime=10, csc_sfact=1e5, csc_fpath=None):
        self.datapath = datapath
        self.brefpath = brefpath

        self.raw = None
        self.bipolar_raw = None
        self.raw_offst = None
        self.sfreq = None
        self.t_init = None
        self.train_start = None
        self.filt_l_freq = filt_l_f
        self.filt_h_freq = filt_h_f
        self.t_vis = t_vis
        self.csc_ftime = csc_ftime
        self.csc_sfactor = csc_sfact
        self.csc_fpath = csc_fpath

        self.load_edf()
        self.delete_raw_annotations()
        self.filter_raw()
        self.bipolar_raw = self.bipolar_eeg_reference()
        # self.set_sleep_annotations()

        self.cdl = BatchCDL(n_atoms=10, n_times_atom=int(self.sfreq / 2),
                            D_init='chunk')

    def bipolar_eeg_reference(self):
        """ Given the montage as defined in _load_montage, we generate the bipolar signal by subtracting the activities in the channels.
        -----------------------------
        Parameters:
            raw, mne object
        -----------------------------
        Returns:
            raw, in the bipolar setting
        """
        print('Setting bipolar reference...')
        reference, ch_names_diff = self.load_eeg_reference()
        data, times = self.raw[:, :]
        ch_names = np.array(self.raw.info['ch_names'])
        new_data = np.zeros((reference.shape[1], data.shape[1]))  # differential measures, time length
        for j, (i_, e_) in enumerate(zip(reference[0], reference[1])):  # iterate on each differential channel
            new_data[j] = (data[np.argwhere(ch_names == i_).squeeze()] -
                           data[np.argwhere(ch_names == e_).squeeze()])
        info_ = mne.create_info(ch_names=list(ch_names_diff),
                                ch_types='eeg', sfreq=self.raw.info['sfreq'])
        info_['meas_date'] = self.raw.info['meas_date']
        print('[Done.]')
        return mne.io.RawArray(data=new_data, info=info_)

    def delete_raw_annotations(self):
        if len(self.raw.annotations) > 0:
            self.raw.annotations.delete(np.arange(len(self.raw.annotations)))

    def filter_raw(self):
        if self.raw is None:
            raise ValueError('Load the raw file first!!!')
        else:
            self.raw = self.raw.filter(self.filt_l_freq, self.filt_h_freq, n_jobs=2,
                                       fir_design='firwin')

    def fit_csc(self):
        data_train, times_train = self.bipolar_raw[:, :int(self.sfreq * self.csc_ftime)]
        data_train *= self.csc_sfactor
        n_ch, n_points = data_train.shape
        print('Fitting {0} second of data...'.format(self.csc_ftime))
        print('This will take some time. Please be patient.')
        self.cdl.fit(data_train.reshape(1, n_ch, n_points))
        print('[Done.]')

    def iterative_activations(self, z_hat=None, tol=1e-3):
        """ Given the dictionary we iteratively find the activations and cluster
            together those which happen in vicinity. It is done in such a way that
            cdl is the dictionary (temporal and spatial pattern + activation for the fitted data.)
            It can also be used for new test data, in a sparse coding approach.
        -----------------------------
        Parameters:
            cdl, the dictionary
            z_hat, the code, by defaultl None if we want to use this on the training data,
                   else the sparse code generated from cdl.transform must be used
            tol, tolerance we go on until we have activations > 1e-3
        -----------------------------
        Returns:
            n_events, n_times
        """
        atom_len = self.cdl.v_hat_.shape[1]
        atom_len *= 2
        n_events = list()
        act_times = list()
        atom_peaks = list()
        if z_hat is None:
            z_hat = self.cdl.z_hat_[0]

        for i_act_, act_ in enumerate(z_hat):
            max_amp_atom = np.argmax(np.abs(self.cdl.v_hat_[i_act_]))
            atom_peaks.append(max_amp_atom)
            binned_act = np.zeros(act_.size)
            act_copy = act_.copy()
            while np.sum(act_copy) > tol:
                id_ = np.argmax(act_copy)
                start = id_ - atom_len // 2
                end = id_ + atom_len // 2
                if start < 0:
                    start = 0
                elif end > act_copy.size:
                    end = act_copy.size
                binned_act[id_] = np.sum(act_copy[start:end])
                act_copy[start:end] = np.zeros(end - start)
            n_events.append(np.sum(binned_act > 0))
            act_times.append(np.squeeze(np.argwhere(binned_act > 0)))

        self.cdl.act_times = act_times
        self.cdl.n_events = n_events
        self.cdl.atom_peaks = atom_peaks

    def select_atoms(self):
        n_atoms = self.cdl.v_hat_.shape[0]
        continue_flag = True
        list_ = list()

        for _a in range(n_atoms):
            atom_events = np.asarray([_e for _e in self.cdl.events if _e[2]==_a])
            self.raw.plot(scalings=500e-6, start=self.train_start, duration=self.t_vis,
                          show=True, events=atom_events, block=True)

        while continue_flag:
            id_good = input(
                "INSERT the INDEX [0, n_atoms-1] correspondent to the candidate pattern. If you want to STOP press S. ")
            if id_good == 'S':
                continue_flag = False
            else:
                try:
                    tmp_good = int(id_good)
                    if tmp_good > n_atoms:
                        print('The index does not correspond to any atom')
                    else:
                        list_.append(tmp_good)
                except:
                    print('Bad integer inserted')
        self.cdl.selected_atoms = list_

        selected_events = np.asarray([_e for _e in self.cdl.events if _e[2] in self.cdl.selected_atoms])
        self.raw.plot(scalings=500e-6, start=self.train_start, duration=self.t_vis,
                      show=True, events=selected_events, block=True)



    def set_events(self, sel_at=None, train=False):
        events = list()
        if sel_at is None:
            sel_at = range(self.cdl.v_hat_.shape[0])
        for _a in sel_at:
            for k_ in self.cdl.act_times[_a]:
                if train:
                    events.append([k_ + self.cdl.atom_peaks[_a] + self.train_start, 0, _a])
                else:
                    events.append([k_ + self.cdl.atom_peaks[_a], 0, _a])
        self.cdl.events = np.asarray(events)

    def load_csc(self, fpath):
        self.cdl = pickle.load(open(fpath, 'rb'), encoding='latin1')

    def load_edf(self):
        self.raw = mne.io.read_raw_edf(self.datapath, preload=True)
        self.raw_offst = self.raw.first_samp
        self.sfreq = self.raw.info['sfreq']
        self.t_init = self.raw.info['meas_date']

    def load_eeg_reference(self):
        """ Here we give the montage through a csv file.
        We use the standard montage for 16 channels"""
        bref = pd.read_csv(self.brefpath, header=None).values
        idx_i = ['EEG ' + ch_ for ch_ in bref[:, 1]]
        idx_e = ['EEG ' + ch_ for ch_ in bref[:, 2]]
        return np.array([idx_i, idx_e]), bref[:, 0]

    def plot_bipolar_raw(self):
        if self.bipolar_raw is None:
            raise ValueError('Load the raw file and set the bipolar reference first!!!')
        else:
            self.bipolar_raw.plot(scalings=500e-6, duration=self.t_vis)
            plt.show()

    def plot_raw(self):
        if self.raw is None:
            raise ValueError('Load the raw file first!!!')
        else:
            self.raw.plot(show_first_samp=True, scalings=dict(eeg=500e-6))
            plt.show()

    def set_sleep_annotations(self):
        sleep_stages = ['Wake', 'N1', 'N2', 'N3', 'REM']
        _ann = mne.Annotations(np.array([0, 0.01, 0.02, 0.03, 0.04], dtype=np.float64),
                               np.array([0.001, 0.001, 0.001, 0.001, 0.001], dtype=np.float64), sleep_stages)
        self.bipolar_raw.set_annotations(_ann)

    def set_train_window(self):
        train_init = int(input("Tell me the starting time: "))
        self.train_start = int(train_init * self.sfreq)

    def main(self):
        self.plot_bipolar_raw()
        if op.isfile(self.csc_fpath):
            self.train_start = 0 #### DA CAMBIARE!!!!!!
            self.load_csc(self.csc_fpath)
        else:
            self.set_train_window()
            self.fit_csc()
        self.iterative_activations()
        self.set_events(train=True)
        self.select_atoms()





if __name__ == '__main__':

    # FOLDERS
    data_folder = '/home/gv/Codici/Python Scripts/DATI_LINO/Dati/'

    # FILENAMES
    data_fname = 'DONZELLAPIETRO.edf'
    bref_fname = 'standard_montage.csv'

    csc_filename = '/home/gv/Codici/Python Scripts/DATI_LINO/Dati/cdl_donzellapietro.pkl'


    # PARAMETERS
    filter_l_freq = 0.1
    filter_h_freq = 40.0
    time_visualization = 30 # seconds
    csc_fit_time = 30  # seconds
    csc_scaling_factor = 1e5

    # PATHS
    data_fpath = op.join(data_folder, data_fname)
    bref_fpath = op.join(data_folder, bref_fname)


    _sc = SpikeCounter(data_fpath, bref_fpath,  filt_l_f=filter_l_freq, filt_h_f=filter_h_freq,
                       csc_ftime=csc_fit_time, csc_sfact=csc_scaling_factor, csc_fpath=csc_filename)

    _sc.main()