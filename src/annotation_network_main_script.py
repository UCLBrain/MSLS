# --------------------------------------------------
#
#     Copyright (C) {2020}  {Kevin Bronik and Le Zhang}
#
#     UCL Medical Physics and Biomedical Engineering
#     https://www.ucl.ac.uk/medical-physics-biomedical-engineering/
#     UCL Queen Square Institute of Neurology
#     https://www.ucl.ac.uk/ion/

#     This program is free software: you can redistribute it and/or modify
#     it under the terms of the GNU General Public License as published by
#     the Free Software Foundation, either version 3 of the License, or
#     (at your option) any later version.
#
#     This program is distributed in the hope that it will be useful,
#     but WITHOUT ANY WARRANTY; without even the implied warranty of
#     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#     GNU General Public License for more details.
#
#     You should have received a copy of the GNU General Public License
#     along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
#
#
#     {Multi-Label Multi/Single-Class Image Segmentation}  Copyright (C) {2020}
#     This program comes with ABSOLUTELY NO WARRANTY; for details type `show w'.
#     This is free software, and you are welcome to redistribute it
#     under certain conditions; type `show c' for details.

# This program uses piece of source code from:
# Title: nicMSlesions
# Author: Sergi Valverde
# Date: 2017
# Code version: 0.2
# Availability: https://github.com/NIC-VICOROB/nicMSlesions

# --------------------------------------------------


import os
import signal
import time
import numpy as np
from nibabel import load as load_nii
import nibabel as nib
from operator import itemgetter
from .annotation_network_build_model import redefine_network_layers_for_training, fit_multifunctional_model, fit_thismodel
from operator import add
from Mkeras.models import load_model
import tensorflow as tf
import configparser

CSELECTED = '\33[7m'
CRED2 = '\33[91m'
CEND = '\33[0m'
CBLINK2 = '\33[6m'
CGREEN  = '\33[32m'

def calcul_num_sample_multi(x_train, settings):

    this_size = len(settings['all_mod'])
    # train_val = {}
    # for label in range(0, this_size):
    #     # perm_indices = np.random.permutation(x_train[label].shape[0])
    #     # train_val[label] = int(len(perm_indices) * train_split_perc)
    #     train_val[label] = int(x_train[label].shape[0])
    #
    # x_train_ = {}
    # train_val = min(train_val.items(), key=lambda x: x[1])
    # train_val = train_val[1]
    #
    # for i in range(0, this_size):
    #     # print("x_train[", i, "]:", x_train[i].shape[0])
    #     x_train_[i] = x_train[i][:train_val]

    temp = {}
    for label in range(0, this_size):
        temp[label] = x_train[label].shape[0]
    # print("temp[label]",label, temp[label])

    num_samples_t = min(temp.items(), key=lambda x: x[1])
    num_samples = num_samples_t[1]

    return num_samples

def training_models(model, train_x_data, train_y_data, settings, thispath):

    # ----------
    # CNN_GUI1
    # ----------

    default_config = configparser.ConfigParser()
    default_config.read(os.path.join(thispath, 'config', 'configuration.cfg'))



    print(CSELECTED + "CNN_GUI: loading training data for first model" +  CEND)
    #modeltest = fit_thismodel(model[0], X, Y, settings)
    X = {}
    Y = {}
    X_val = {}
    Y_val = {}
    y_data = {}
    scans = list(train_x_data.keys())
    # label_n = ['LB1', 'LB2', 'LB3', 'LB4', 'LB5']
    sel_voxels_train = {}
    sel_voxels_val = {}
    # train_y_data = {f: os.path.join(settings['train_folder'],

    this_size = len(settings['all_isolated_label'])


    for n, i in zip(range(1, this_size + 1), range(0, this_size)):
         # y_data[i] = [train_y_data[s][n] for s in scans]
         y_data[i] = {s: train_y_data[s][n] for s in scans}


    label_n = settings['label_mods']
    max_class = int(this_size / 5)
    if settings['ms_dataset']:
        # for i in range(0, 5):
        for j in range(0, max_class):
            for i in range(0, 5):
                print("Loading data of label:", CRED2 + label_n[i] + CEND, ", class:", CGREEN + str(j + 1) + CEND, "for training")
                # print(y_data[i * max_class + j])
                X[i + 5 * j], Y[i + 5 * j], sel_voxels_train[i + 5 * j] = load_data_for_training(train_x_data, y_data[i * max_class + j], settings, False)
                print("Loading data of label:", CRED2 + label_n[i] + CEND, ", class:", CGREEN + str(j + 1) + CEND, "for cross validation")
                X_val[i + 5 * j], Y_val[i + 5 * j], sel_voxels_val[i + 5 * j] = load_data_for_training(train_x_data, y_data[i * max_class + j],
                                                                                                   settings, False, set_this_value=settings['fract_negative_positive_CV'])

    if settings['mnist_dataset']:
        # for i in range(0, 5):
        for j in range(0, max_class):
            for i in range(0, 5):
                print("Loading data of label:", CRED2 + label_n[i] + CEND, ", class:", CGREEN + str(j + 1) + CEND, "for training")
                # print(y_data[i * max_class + j])
                X[i + 5 * j], Y[i + 5 * j], sel_voxels_train[i + 5 * j] = load_data_for_training(train_x_data, y_data[i * max_class + j], settings, False)
                print("Loading data of label:", CRED2 + label_n[i] + CEND, ", class:", CGREEN + str(j + 1) + CEND, "for cross validation")
                # print(y_data[i * max_class + j])
                X_val[i + 5 * j], Y_val[i + 5 * j], sel_voxels_val[i + 5 * j] = load_data_for_training(train_x_data, y_data[i * max_class + j],
                                                                                                   settings, False, set_this_value=settings['fract_negative_positive_CV'])

    net_model_name = model[0]['special_name_1']
    net_model_name_2 = model[1]['special_name_2']


    if settings['full_train'] is False:
        max_epochs = settings['max_epochs']
        patience = 0
        best_val_loss = np.Inf
        numsamplex = int(calcul_num_sample_multi(X, settings))
        model[0] = redefine_network_layers_for_training(model=model[0], settings=settings['all_isolated_label'],
                                          num_layers=settings['num_layers'],
                                          number_of_samples=numsamplex)
        settings['max_epochs'] = 0
        for it in range(0, max_epochs, 10):
            settings['max_epochs'] += 10
            # model[0] = fit_multifunctional_model(model[0], X, Y, settings,
            #                      initial_epoch=it)
            model[0] = fit_multifunctional_model(model[0], X, Y, settings, X_val, Y_val, initial_epoch=it)

            # evaluate if continuing training or not
            val_loss = min(model[0]['history'].history['val_loss'])
            if val_loss > best_val_loss:
                patience += 10
            else:
                best_val_loss = val_loss

            if patience >= settings['patience']:
                break

            if settings['ms_dataset']:
                # for i in range(0, 5):
                for j in range(0, max_class):
                    for i in range(0, 5):
                        print("Loading data of label:", CRED2 + label_n[i] + CEND, ", class:",
                              CGREEN + str(j + 1) + CEND, "for training")
                        # print(y_data[i * max_class + j])
                        X[i + 5 * j], Y[i + 5 * j], sel_voxels_train[i + 5 * j] = load_data_for_training(train_x_data,
                                                                                                     y_data[i * max_class + j],
                                                                                                     settings, False)
                        print("Loading data of label:", CRED2 + label_n[i] + CEND, ", class:",
                              CGREEN + str(j + 1) + CEND, "for cross validation")
                        X_val[i + 5 * j], Y_val[i + 5 * j], sel_voxels_val[i + 5 * j] = load_data_for_training(train_x_data,
                                                                                                           y_data[i * max_class + j],
                                                                                                           settings,
                                                                                                           False,
                                                                                                           set_this_value=settings['fract_negative_positive_CV'])

            if settings['mnist_dataset']:
                # for i in range(0, 5):
                for j in range(0, max_class):
                    for i in range(0, 5):
                        print("Loading data of label:", CRED2 + label_n[i] + CEND, ", class:",
                              CGREEN + str(j + 1) + CEND, "for training")
                        # print(y_data[i * max_class + j])
                        X[i + 5 * j], Y[i + 5 * j], sel_voxels_train[i + 5 * j] = load_data_for_training(train_x_data,
                                                                                                     y_data[i * max_class + j],
                                                                                                     settings, False)
                        print("Loading data of label:", CRED2 + label_n[i] + CEND, ", class:",
                              CGREEN + str(j + 1) + CEND, "for cross validation")
                        # print(y_data[i * max_class + j])
                        X_val[i + 5 * j], Y_val[i + 5 * j], sel_voxels_val[i + 5 * j] = load_data_for_training(train_x_data,
                                                                                                           y_data[i * max_class + j],
                                                                                                           settings,
                                                                                                           False,
                                                                                                           set_this_value=settings['fract_negative_positive_CV'])


        settings['max_epochs'] = max_epochs
    else:
        # model[0] = load_model(net_weights_1)
        # net_model_name = model[0]['special_name_1']
        if os.path.exists(os.path.join(settings['weight_paths'], settings['model_name'],'nets', net_model_name + '.hdf5')) and \
                not os.path.exists(os.path.join(settings['weight_paths'], settings['model_name'],'nets', net_model_name_2 + '.hdf5')):
            net_weights_1 = os.path.join(settings['weight_paths'], settings['model_name'],'nets', net_model_name + '.hdf5')
            try:
                model[0]['net'].load_weights(net_weights_1, by_name=True)
                print("CNN_GUI has Loaded previous weights from the", net_weights_1)
            except:
                print("> ERROR: The model", \
                    settings['model_name'], \
                    'selected does not contain a valid network model')
                time.sleep(1)
                os.kill(os.getpid(), signal.SIGTERM)

        if not os.path.exists(os.path.join(settings['weight_paths'], settings['model_name'],'nets', net_model_name_2 + '.hdf5')) and \
                (settings['model_1_train'] is  False):
          model[0] = fit_multifunctional_model(model[0], X, Y, settings, X_val, Y_val)
          default_config.set('completed', 'model_1_train', str(True))
          with open(os.path.join(thispath,
                                 'config',
                                 'configuration.cfg'), 'w') as configfile:
              default_config.write(configfile)
          settings['model_1_train'] = True
        # thismodel = os.path.join(CURRENT_PATH
        #                        , 'SAVEDMODEL', net_model_name + '.h5')
        # model[0]['net'].save(thismodel)
        M1 = default_config.get('completed', 'model_1_train')
        print('Was first model created successfully?', M1)
    # only if No cascaded model
    # if settings['model_1_train'] is True:
    #     print 'No cascaded model: Only model one has been created'
    #     return model[0]
    # ----------
    # CNN_GUI2
    # ----------

    print(CSELECTED +"CNN_GUI: loading training data for the second model"+ CEND)
    if settings['model_2_train'] is False:
        if settings['ms_dataset']:
            # for i in range(0, 5):
            for j in range(0, max_class):
                for i in range(0, 5):
                    print("Loading data of label:", CRED2 + label_n[i] + CEND, ", class:", CGREEN + str(j + 1) + CEND,
                          "for training")
                    # print(y_data[i * max_class + j])
                    X[i + 5 * j], Y[i + 5 * j], sel_voxels_train[i + 5 * j] = load_data_for_training(train_x_data, y_data[
                        i * max_class + j], settings, False, model=model[0], index=i + 5 * j)
                    print("Loading data of label:", CRED2 + label_n[i] + CEND, ", class:", CGREEN + str(j + 1) + CEND,
                          "for cross validation")
                    X_val[i + 5 * j], Y_val[i + 5 * j], sel_voxels_val[i + 5 * j] = load_data_for_training(train_x_data,
                                                                                                       y_data[i * max_class + j],
                                                                                                       settings, False,
                                                                                                       set_this_value=settings['fract_negative_positive_CV'],
                                                                                                       model=model[0], index=i + 5 * j)

        if settings['mnist_dataset']:
            # for i in range(0, 5):
            for j in range(0, max_class):
                for i in range(0, 5):
                    print("Loading data of label:", CRED2 + label_n[i] + CEND, ", class:", CGREEN + str(j + 1) + CEND,
                          "for training")
                    # print(y_data[i * max_class + j])
                    X[i + 5 * j], Y[i + 5 * j], sel_voxels_train[i + 5 * j] = load_data_for_training(train_x_data, y_data[
                        i * max_class + j], settings, False, model=model[0], index=i + 5 * j)
                    print("Loading data of label:", CRED2 + label_n[i] + CEND, ", class:", CGREEN + str(j + 1) + CEND,
                          "for cross validation")
                    # print(y_data[i * max_class + j])
                    X_val[i + 5 * j], Y_val[i + 5 * j], sel_voxels_val[i + 5 * j] = load_data_for_training(train_x_data,
                                                                                                       y_data[i * max_class + j],
                                                                                                       settings, False,
                                                                                                       set_this_value=settings['fract_negative_positive_CV'],
                                                                                                       model=model[0],
                                                                                                       index=i + 5 * j)

      # print('> CNN_GUI: train_x ', X.shape)

    # define training layers
    if settings['full_train'] is False:
        max_epochs = settings['max_epochs']
        patience = 0
        best_val_loss = np.Inf
        numsamplex = int(calcul_num_sample_multi(X, settings))
        model[1] = redefine_network_layers_for_training(model=model[1], settings=settings['all_isolated_label'],
                                          num_layers=settings['num_layers'],
                                          number_of_samples=numsamplex)

        settings['max_epochs'] = 0
        for it in range(0, max_epochs, 10):
            settings['max_epochs'] += 10
            # model[1] = fit_multifunctional_model(model[1], X, Y, settings,
            #                      initial_epoch=it)

            model[1] = fit_multifunctional_model(model[1], X, Y, settings, X_val, Y_val, initial_epoch=it)

            # evaluate if continuing training or not
            val_loss = min(model[0]['history'].history['val_loss'])
            if val_loss > best_val_loss:
                patience += 10
            else:
                best_val_loss = val_loss

            if patience >= settings['patience']:
                break


            if settings['ms_dataset']:
                # for i in range(0, 5):
                for j in range(0, max_class):
                    for i in range(0, 5):
                        print("Loading data of label:", CRED2 + label_n[i] + CEND, ", class:",
                              CGREEN + str(j + 1) + CEND,
                              "for training")
                        # print(y_data[i * max_class + j])
                        X[i + 5 * j], Y[i + 5 * j], sel_voxels_train[i + 5 * j] = load_data_for_training(train_x_data,
                                                                                                     y_data[i * max_class + j],
                                                                                                     settings, False,
                                                                                                     model=model[0],
                                                                                                     selected_voxels=sel_voxels_train[i + 5 * j], index=i + 5 * j)
                        print("Loading data of label:", CRED2 + label_n[i] + CEND, ", class:",
                              CGREEN + str(j + 1) + CEND,
                              "for cross validation")
                        X_val[i + 5 * j], Y_val[i + 5 * j], sel_voxels_val[i + 5 * j] = load_data_for_training(train_x_data,
                                                                                                           y_data[
                                                                                                               i * max_class + j],
                                                                                                           settings,
                                                                                                           False,
                                                                                                           set_this_value=settings['fract_negative_positive_CV'],
                                                                                                           model=model[0],
                                                                                                           selected_voxels=sel_voxels_train[i + 5 * j], index=i + 5 * j)

            if settings['mnist_dataset']:
                # for i in range(0, 5):
                for j in range(0, max_class):
                    for i in range(0, 5):
                        print("Loading data of label:", CRED2 + label_n[i] + CEND, ", class:",
                              CGREEN + str(j + 1) + CEND,
                              "for training")
                        # print(y_data[i * max_class + j])
                        X[i + 5 * j], Y[i + 5 * j], sel_voxels_train[i + 5 * j] = load_data_for_training(train_x_data,
                                                                                                     y_data[
                                                                                                         i * max_class + j],
                                                                                                     settings, False,
                                                                                                     model=model[0],
                                                                                                     selected_voxels=sel_voxels_train[i + 5 * j], index=i + 5 * j)
                        print("Loading data of label:", CRED2 + label_n[i] + CEND, ", class:",
                              CGREEN + str(j + 1) + CEND,
                              "for cross validation")
                        # print(y_data[i * max_class + j])
                        X_val[i + 5 * j], Y_val[i + 5 * j], sel_voxels_val[i + 5 * j] = load_data_for_training(train_x_data,
                                                                                                           y_data[
                                                                                                               i * max_class + j],
                                                                                                           settings,
                                                                                                           False,
                                                                                                           set_this_value=settings['fract_negative_positive_CV'],
                                                                                                           model=model[0],
                                                                                                           selected_voxels=sel_voxels_train[i + 5 * j], index=i + 5 * j)


        settings['max_epochs'] = max_epochs
    else:
        # model[1] = fit_multifunctional_model(model[1], X, Y, settings)
        if os.path.exists(os.path.join(settings['weight_paths'], settings['model_name'],'nets', net_model_name + '.hdf5'))  and  \
                os.path.exists(os.path.join(settings['weight_paths'], settings['model_name'],'nets', net_model_name_2 + '.hdf5')):
            net_weights_2 = os.path.join(settings['weight_paths'], settings['model_name'],'nets', net_model_name_2 + '.hdf5')
            try:
                model[1]['net'].load_weights(net_weights_2, by_name=True)
                print("CNN_GUI has Loaded previous weights from the", net_weights_2)
            except:
                print("> ERROR: The model", \
                    settings['model_name'], \
                    'selected does not contain a valid network model')
                time.sleep(1)
                os.kill(os.getpid(), signal.SIGTERM)

        if os.path.exists(os.path.join(settings['weight_paths'], settings['model_name'],'nets', net_model_name + '.hdf5')) and settings['model_1_train'] \
                and (settings['model_2_train'] is False):
          # model[1] = fit_multifunctional_model(model[1], X, Y, settings)
          model[1] = fit_multifunctional_model(model[1], X, Y, settings, X_val, Y_val)
          default_config.set('completed', 'model_2_train', str(True))
          with open(os.path.join(thispath,
                                 'config',
                                 'configuration.cfg'), 'w') as configfile:
              default_config.write(configfile)
          settings['model_2_train'] = True
        M2 = default_config.get('completed', 'model_2_train')
        print('Was second model created successfully?', M2)


    return model


def testing_models(model, test_x_data, settings):


    # print '> CNN_GUI: testing the model'
    labels = settings['label_mods']
    this_size = len(settings['all_isolated_label'])
    n_class = int(this_size / 5)
    label_n = settings['label_mods']
    # organize model_names
    exp_folder = os.path.join(settings['test_folder'],
                              settings['run_testing'],
                              settings['model_name'])
    if not os.path.exists(exp_folder):
        os.mkdir(exp_folder)

    # first network
    firstnetwork_time = time.time()
    first_model_seg = {}
    print('\x1b[6;30;41m' + 'Prediction of first model is started ...' + '\x1b[0m')
    # settings['test_name'] = 'First_model_probability_map.nii.gz'
    for i in range(0, this_size):
        label_nr = i
        save_nifti = True if settings['debug'] is True else False
        print(CSELECTED + "First model, input:", str(i + 1), "probability map" + CEND)
        print("Input image(s):", str(test_x_data))
        # settings['test_name'] = 'First_model_class_' + str(int(i / 5) + 1) + '_LB' + str(int(i + 1)) +'_probability_map.nii.gz'
        if i >= 5:
            label_nr = label_nr - 5 * int(i / 5)
            settings['test_name'] = 'First_model_class_' + str(int(i / 5) + 1) + '_[' + label_n[label_nr] + ']_probability_map.nii.gz'
        else:
            settings['test_name'] = 'First_model_class_' + str(int(i / 5) + 1) + '_[' + label_n[
                label_nr] + ']_probability_map.nii.gz'
        # print(CSELECTED +"First model, label:", labels[i], "probability map"+ CEND)
        first_model_seg[i] = run_testing(model[0],
                              test_x_data,
                              settings,
                              index=i,
                              save_nifti=save_nifti)

    print("> INFO:............",  "total pipeline time for first network ", round(time.time() - firstnetwork_time), "sec")

    # # second network
    secondnetwork_time = time.time()
    second_model_seg = {}
    Cvoxel = {}
    print('\x1b[6;30;41m' + 'Prediction of second model is started ...' + '\x1b[0m')
    for i in range(0, this_size):
        #Cvoxel[i] = first_model_seg[i] > 0.5  # <.... tested work well
        Cvoxel[i] = first_model_seg[i] > 0.4
        if np.sum(Cvoxel[i]) == 1 or np.sum(Cvoxel[i]) == 0:
            Cvoxel[i] = first_model_seg[i] > 0
        # Cvoxel[i] = first_model_seg[i] > np.mean(first_model_seg[i])
    # settings['test_name'] = 'Second_model_probability_map.nii.gz'
    for i in range(0, this_size):
        label_nr = i
        print(CSELECTED + "Second model, input:", str(i), "probability map" + CEND)
        print("Input image(s):", str(test_x_data))
        # settings['test_name'] = 'Second_model_class_' + str(int(i / 5) + 1) + '_LB' + str(int(i + 1)) +'_probability_map.nii.gz'
        if i >= 5:
            label_nr = label_nr - 5 * int(i / 5)
            settings['test_name'] = 'Second_model_class_' + str(int(i / 5) + 1) + '_[' + label_n[label_nr] + ']_probability_map.nii.gz'
        else:
            settings['test_name'] = 'Second_model_class_' + str(int(i / 5) + 1) + '_[' + label_n[
                label_nr] + ']_probability_map.nii.gz'
        second_model_seg[i] = run_testing(model[1],
                              test_x_data,
                              settings,
                              index=i,
                              save_nifti=True,
                              candidate_mask=Cvoxel)

    print("> INFO:............", "total pipeline time for second  network", round(time.time() - secondnetwork_time),
          "sec")

    # postprocess the output segmentation
    # obtain the orientation from the first scan used for testing
    scans = list(test_x_data.keys())
    flair_scans = [test_x_data[s]['FLAIR'] for s in scans]
    flair_image = load_nii(flair_scans[0])
    #settings['test_name'] = settings['model_name'] + '_hard_seg.nii.gz'

    print('\x1b[6;30;41m' + 'Prediction single class segmentation is started ...' + '\x1b[0m')

    # settings['test_name'] = 'CNN_GUI_final_single_class_segmentation.nii.gz'
    for i in range(0, this_size):
        label_nr = i
        if i >= 5:
            label_nr = label_nr - 5 * int(i / 5)
            print(CSELECTED + "Final single class segmentation,", "class:", str(int(i / 5) + 1) + CEND, " input label:",
                  CRED2 + label_n[label_nr] + CEND)
            # settings['test_name'] = 'CNN_final_single_class_' + str(int(i / 5) + 1) + '_LB' + str(int(i + 1)) + '_segmentation.nii.gz'
            settings['test_name'] = 'CNN_final_single_class_' + str(int(i / 5) + 1) + '_[' + label_n[label_nr] + ']_probability_map.nii.gz'
        else:
            print(CSELECTED + "Final single class segmentation,", "class:", str(int(i / 5) + 1) + CEND,
                  " input label:",
                  CRED2 + label_n[label_nr] + CEND)
            # settings['test_name'] = 'CNN_final_single_class_' + str(int(i / 5) + 1) + '_LB' + str(int(i + 1)) + '_segmentation.nii.gz'
            settings['test_name'] = 'CNN_final_single_class_' + str(int(i / 5) + 1) + '_[' + label_n[
                label_nr] + ']_probability_map.nii.gz'

        segmentation = segmentation_final_process(second_model_seg[i],
                                                     settings,
                                                     save_nifti=True,
                                                     orientation=flair_image.affine)

    print('')
    if this_size > 5:
        for j in range(0, 5):
            # settings['test_name'] = 'CNN_final_multi_class_' + '_LB' + str(
            #         int(j + 1)) + '_segmentation.nii.gz'
            settings['test_name'] = 'CNN_final_multi_class_' + '_[' + label_n[j] + ']_segmentation.nii.gz'
            output_scan_multi = np.zeros_like(second_model_seg[0])
            seg_im = {}

            for i in range(0, n_class):
                # file_n = 'CNN_final_single_class_' + str(int(i / 5) + 1) + '_LB' + str(
                #     int(j + 1)) + '_segmentation.nii.gz'
                file_n = 'CNN_final_single_class_' + str(i + 1) + '_[' + label_n[j] + ']_probability_map.nii.gz'
                image = nib.load(
                    os.path.join(settings['test_folder'], settings['run_testing'], settings['model_name'], file_n))
                seg_im[i] = image.get_data()

            for i in range(0, n_class):
                current_voxels = np.stack(np.where(seg_im[i] == 1), axis=1)
                output_scan_multi[current_voxels[:, 0], current_voxels[:, 1], current_voxels[:, 2]] = 1 + i

            print('\x1b[6;30;41m' + 'Prediction final step (multi class segmentation) is started ...' + '\x1b[0m')
            nifti_out = nib.Nifti1Image(output_scan_multi,
                                        affine=flair_image.affine)
            nifti_out.to_filename(os.path.join(settings['test_folder'],
                                               settings['run_testing'],
                                               settings['model_name'],
                                               settings['test_name']))

    print('')
    print('\x1b[6;30;41m' + 'Prediction is done!' + '\x1b[0m')
    # return out_segmentation
    return segmentation


def normalize_data(im, datatype=np.float32):
    """
    zero mean / 1 standard deviation image normalization

    """
    im = im.astype(dtype=datatype) - im[np.nonzero(im)].mean()
    im = im / im[np.nonzero(im)].std()

    return im


def select_voxels(input_masks, settings,  threshold=2, datatype=np.float32):

    check_ms_dataset = settings['ms_dataset']
    check_mnist_dataset = settings['mnist_dataset']
    # load images and normalize their intensities
    images = [load_nii(image_name).get_data() for image_name in input_masks]
    images_norm = [normalize_data(im) for im in images]
    # select voxels with intensity higher than threshold

    # for ms data please change this rois = [image > 0.5 for image in images_norm]
    rois = []

    if check_ms_dataset:
        rois = [image > 0.5 for image in images_norm]

    if check_mnist_dataset:
        rois = [image > -2.5 for image in images_norm]

    return rois

def load_data_for_training(train_x_data,
                       train_y_data,
                       settings,
                       check,
                       model=None,
                       selected_voxels=None, index=0, set_this_value=0.0):


    # get_scan names and number of modalities used
    scans = list(train_x_data.keys())
    modalities = list(train_x_data[scans[0]].keys())
    # flair_scans = [train_x_data[s]['FLAIR'] for s in scans]
    # select voxels for training:
    #  if model is no passed, training samples are extract by discarding CSF
    #  and darker WM in FLAIR, and use all remaining voxels.
    #  if model is passes, use the trained model to extract all voxels
    #  with probability > 0.5
    if model is None:
        flair_scans = [train_x_data[s]['FLAIR'] for s in scans]
        selected_voxels = select_voxels(flair_scans, settings,
                                                 settings['min_th'])
    elif selected_voxels is None:
        selected_voxels = thresholded_voxels_from_learned_model(model,
                                                            train_x_data,
                                                            settings, index)
    else:
        pass
    # extract patches and labels for each of the modalities
    data = []

    for m in modalities:
        x_data = [train_x_data[s][m] for s in scans]
        y_data = [train_y_data[s] for s in scans]
        x_patches, y_patches = train_patch_to_tensor(x_data,
                                                  y_data,
                                                  selected_voxels,
                                                  settings['patch_size'],
                                                  settings['balanced_training'],
                                                  settings['fract_negative_positive'], check, settings, set=set_this_value)
        data.append(x_patches)

    # stack patches in channels [samples, channels, p1, p2, p3]
    X = np.stack(data, axis=1)
    Y = y_patches

    # apply randomization if selected
    if settings['randomize_train']:

        seed = np.random.randint(np.iinfo(np.int32).max)
        np.random.seed(seed)
        X = np.random.permutation(X.astype(dtype=np.float32))
        np.random.seed(seed)
        Y = np.random.permutation(Y.astype(dtype=np.int32))

    # fully convolutional / voxel labels
    if settings['fully_convolutional']:
        # Y = [ num_samples, 1, p1, p2, p3]
        Y = np.expand_dims(Y, axis=1)
    else:
        # Y = [num_samples,]
        if Y.shape[3] == 1:
            Y = Y[:, Y.shape[1] // 2, Y.shape[2] // 2, :]
        else:
            Y = Y[:, Y.shape[1] // 2, Y.shape[2] // 2, Y.shape[3] // 2]
        Y = np.squeeze(Y)

    return X, Y, selected_voxels

def train_patch_to_tensor(x_data,
                       y_data,
                       selected_voxels,
                       patch_size,
                       balanced_training,
                       fraction_negatives,
                       check,
                       settings,
                       random_state=42,
                       datatype=np.float32, set=0.0):
    """
    Load train patches with size equal to patch_size, given a list of
    selected voxels

    Inputs:
       - x_data: list containing all subject image paths for a single modality
       - y_data: list containing all subject image paths for the labels
       - selected_voxels: list where each element contains the subject binary
         mask for selected voxels [len(x), len(y), len(z)]
       - tuple containing patch size, either 2D (p1, p2, 1) or 3D (p1, p2, p3)

    Outputs:
       - X: Train X data matrix for the particular channel
       - Y: Train Y labels [num_samples, p1, p2, p3]
    """

    # load images and normalize their intensties
    images = [load_nii(name).get_data() for name in x_data]
    images_norm = [normalize_data(im) for im in images]

    # load labels testing .....

    #
    # lesion_masks_test = [load_nii(name).get_data()
    #                 for name in y_data]
    # lesion_centers_test = [Compute_voxel_coordinates(mask) for mask in lesion_masks_test]

    lesion_masks = [load_nii(name).get_data().astype(dtype=np.bool)
                    for name in y_data]

    nolesion_masks = [np.logical_and(np.logical_not(lesion), brain)
                      for lesion, brain in zip(lesion_masks, selected_voxels)]

    # Get all the x,y,z coordinates for each image
    for nlm in range(0, len(nolesion_masks)):
        if not nolesion_masks[nlm].any():
            nolesion_masks[nlm] = np.logical_and(np.logical_not(lesion_masks[nlm]), (images_norm[nlm] > -2.5))
            print('\x1b[6;30;41m' + 'Warning:' + '\x1b[0m', 'for the training scan:', x_data[nlm],' after applying probability higher than 0.5, no voxels have been selected as no lesion, using original data instead!' )
            print('')


    lesion_centers = [Compute_voxel_coordinates(mask, settings) for mask in lesion_masks]
    # lesion_centers = [Compute_voxel_coordinates(mask, norm) for mask, norm in zip(lesion_masks, images_norm)]


    nolesion_centers = [Compute_voxel_coordinates(mask, settings) for mask in nolesion_masks]

    # load all positive samples (lesion voxels). If a balanced training is set
    # use the same number of positive and negative samples. On unbalanced
    # training sets, the number of negative samples is multiplied by
    # of random negatives samples

    np.random.seed(random_state)

    #x_pos_patches = [np.array(get_patches(image, centers, patch_size))
    #                 for image, centers in zip(images_norm, lesion_centers)]
    #y_pos_patches = [np.array(get_patches(image, centers, patch_size))
    #                 for image, centers in zip(lesion_masks, lesion_centers)]

    number_lesions = [np.sum(lesion) for lesion in lesion_masks]
    total_lesions = np.sum(number_lesions)
    if set != 0.0:
        fraction_negatives = set
        # print("fraction_negatives new value is:", fraction_negatives)
    else:
        # print("fraction_negatives old value is:", fraction_negatives)
        pass
    neg_samples = int((total_lesions * fraction_negatives) / len(number_lesions))
    X, Y = [], []

    for l_centers, nl_centers, image, lesion in zip(lesion_centers,
                                                    nolesion_centers,
                                                    images_norm,
                                                    lesion_masks):

        # balanced training: same number of positive and negative samples
        # if balanced_training:
        if check:
            if len(l_centers) > 0:
                # positive samples
                x_pos_samples = get_patches(image, l_centers, patch_size)
                y_pos_samples = get_patches(lesion, l_centers, patch_size)
                idx = np.random.permutation(list(range(0, len(nl_centers)))).tolist()[:len(l_centers)]
                nolesion = itemgetter(*idx)(nl_centers)
                x_neg_samples = get_patches(image, nolesion, patch_size)
                y_neg_samples = get_patches(lesion, nolesion, patch_size)
                X.append(np.concatenate([x_pos_samples, x_neg_samples]))
                Y.append(np.concatenate([y_pos_samples, y_neg_samples]))

        # unbalanced dataset: images with only negative samples are allowed
        else:
            if len(l_centers) > 0:
                x_pos_samples = get_patches(image, l_centers, patch_size)
                y_pos_samples = get_patches(lesion, l_centers, patch_size)

            idx = np.random.permutation(list(range(0, len(nl_centers)))).tolist()[:neg_samples]
            nolesion = itemgetter(*idx)(nl_centers)
            x_neg_samples = get_patches(image, nolesion, patch_size)
            y_neg_samples = get_patches(lesion, nolesion, patch_size)

            # concatenate positive and negative samples
            if len(l_centers) > 0:
                X.append(np.concatenate([x_pos_samples, x_neg_samples]))
                Y.append(np.concatenate([y_pos_samples, y_neg_samples]))
            else:
                X.append(x_neg_samples)
                Y.append(y_neg_samples)

    X = np.concatenate(X, axis=0)
    Y = np.concatenate(Y, axis=0)

    return X, Y


def test_patch_to_tensor(test_x_data,
                      patch_size,
                      batch_size,
                      settings,
                      voxel_candidates=None,
                      datatype=np.float32):
    """
    Function generator to load test patches with size equal to patch_size,
    given a list of selected voxels. Patches are returned in batches to reduce
    the amount of RAM used

    Inputs:
       - x_data: list containing all subject image paths for a single modality
       - selected_voxels: list where each element contains the subject binary
         mask for selected voxels [len(x), len(y), len(z)]
       - tuple containing patch size, either 2D (p1, p2, 1) or 3D (p1, p2, p3)
       - Voxel candidates: a binary mask containing voxels for testing

    Outputs (in batches):
       - X: Train X data matrix for the each channel [num_samples, p1, p2, p3]
       - voxel_coord: list of tuples with voxel coordinates (x,y,z) of
         selected patches
    """

    # get scan names and number of modalities used
    scans = list(test_x_data.keys())
    modalities = list(test_x_data[scans[0]].keys())

    # load all image modalities and normalize intensities
    images = []

    for m in modalities:
        raw_images = [load_nii(test_x_data[s][m]).get_data() for s in scans]
        images.append([normalize_data(im) for im in raw_images])

    # select voxels for testing. Discard CSF and darker WM in FLAIR.
    # If voxel_candidates is not selected, using intensity > 0.5 in FLAIR,
    # else use the binary mask to extract candidate voxels
    if voxel_candidates is None:
        flair_scans = [test_x_data[s]['FLAIR'] for s in scans]
        selected_voxels = [Compute_voxel_coordinates_test(mask)
                           for mask in select_voxels(flair_scans, settings,
                                                              0.5)][0]
    else:
        selected_voxels = Compute_voxel_coordinates_test(voxel_candidates)

    # yield data for testing with size equal to batch_size
    # for i in range(0, len(selected_voxels), batch_size):
    #     c_centers = selected_voxels[i:i+batch_size]
    #     X = []
    #     for m, image_modality in zip(modalities, images):
    #         X.append(get_patches(image_modality[0], c_centers, patch_size))
    #     yield np.stack(X, axis=1), c_centers

    X = []

    for image_modality in images:
        X.append(get_patches(image_modality[0], selected_voxels, patch_size))
    # x_ = np.empty((9200, 400, 400, 3)
    # Xs = np.zeros_like (X)
    Xs = np.stack(X, axis=1)
    return Xs, selected_voxels

def sc_one_zero(array):
    for x in array.flat:
        if x!=1 and x!=0:
            return True
    return False

def Compute_voxel_coordinates_test(mask):
    """
    Compute x,y,z coordinates of a binary mask

    Input:
       - mask: binary mask

    Output:
       - list of tuples containing the (x,y,z) coordinate for each of the
         input voxels
    """
    if np.sum(mask) > 0:
       indices = np.stack(np.nonzero(mask), axis=1)
       indices = [tuple(idx) for idx in indices]
    else:
        arr = np.zeros_like(mask)
        indices = np.stack(np.where(arr == 0), axis=1)
        indices = [tuple(idx) for idx in indices]

    return indices

def rand_bin_array(K, N):
    arr = np.zeros_like(N)
    print(arr.shape)
    arr = np.transpose(arr)
    a = int(arr.shape[0] * 1 / 3)
    b = int(arr.shape[1] * 2 / 3)
    arr[a:b:20] = 1
    # np.random.shuffle(arr)
    return arr

def Compute_voxel_coordinates(mask, settings):

    check_ms_dataset = settings['ms_dataset']
    check_mnist_dataset = settings['mnist_dataset']

    indices=[]
    if check_ms_dataset:
       indices = np.stack(np.nonzero(mask), axis=1)
       indices = [tuple(idx) for idx in indices]

    if check_mnist_dataset:

         if np.sum(mask) > 10:
             indices = np.stack(np.nonzero(mask), axis=1)
             indices = [tuple(idx) for idx in indices]

         else:
             arr = np.zeros_like(mask)
             K = np.int32(arr.shape[0] / 2)
             arr[:K] = 1
             indices = np.stack(np.where(arr == 1), axis=1)
             indices = [tuple(idx) for idx in indices]

    return indices


def get_patches(image, centers, patch_size=(15, 15, 15)):
    """
    Get image patches of arbitrary size based on a set of centers
    """
    # If the size has even numbers, the patch will be centered. If not,
    # it will try to create an square almost centered. By doing this we allow
    # pooling when using encoders/unets.
    patches = []
    list_of_tuples = all([isinstance(center, tuple) for center in centers])
    sizes_match = [len(center) == len(patch_size) for center in centers]
    # sizes_match = [len(str(center)) == len(str(patch_size)) for center in centers]

    if list_of_tuples and sizes_match:
        patch_half = tuple([idx//2 for idx in patch_size])
        new_centers = [list(map(add, center, patch_half)) for center in centers]
        padding = tuple((idx, size-idx)
                        for idx, size in zip(patch_half, patch_size))
        new_image = np.pad(image, padding, mode='constant', constant_values=0)
        slices = [[slice(c_idx-p_idx, c_idx+(s_idx-p_idx))
                   for (c_idx, p_idx, s_idx) in zip(center,
                                                    patch_half,
                                                    patch_size)]
                  for center in new_centers]

        # patches = [new_image[idx] for idx in slices]
        patches = [new_image[tuple(idx)] for idx in slices]

    return patches


def run_testing(model,
              test_x_data,
              settings,
              index,
              save_nifti=True,
              candidate_mask=None):


    # get_scan name and create an empty nifti image to store segmentation
    scans = list(test_x_data.keys())
    flair_scans = [test_x_data[s]['FLAIR'] for s in scans]
    flair_image = load_nii(flair_scans[0])
    seg_image = np.zeros_like(flair_image.get_data().astype('float32'))
    this_size = len(settings['all_isolated_label'])
    if candidate_mask is not None:
        all_voxels = np.sum(candidate_mask[index])
    else:
        all_voxels = np.sum(flair_image.get_data() > 0)

    if settings['debug'] is True:
            print("> DEBUG ", scans[0], "Voxels to classify:", all_voxels)

    # compute lesion segmentation in batches of size settings['batch_size']
    # batch, centers = test_patch_to_tensor(test_x_data,
    #                                    settings['patch_size'],
    #                                    settings['batch_size'],
    #                                    candidate_mask)
    batch = {}
    centers = {}
    if candidate_mask is None:

        for i in range(0, this_size):
            batch[i], centers[i] = test_patch_to_tensor(test_x_data,
                                                     settings['patch_size'],
                                                     settings['batch_size'], settings)

    if candidate_mask is not None:

        for i in range(0, this_size):
            batch[i], centers[i] = test_patch_to_tensor(test_x_data,
                                                     settings['patch_size'],
                                                     settings['batch_size'], settings,
                                                     candidate_mask[i])
    # print ("centers:", centers)

    print("Testing [", index + 1, "]:", batch[index].shape, end=' ')

    # if settings['debug'] is True:
    #     print("> DEBUG: testing current_batch:", batch.shape, end=' ')

    batch_array = []
    for ind in range(0, this_size):
        batch_array.append(np.squeeze(batch[index]))

    print(" \n")
    # print("Prediction or loading learned model started........................> \n")
    w_class = int(index / 5) + 1
    out_class = int(index)
    print("Prediction or loading learned model for input:", '\x1b[6;30;41m' + str(index + 1) + '\x1b[0m', ", class:", CGREEN + str(w_class) + CEND, "\n")
    # print("Loading data of label:", CRED2 + label_n[i] + CENprint("Prediction or loading learned model started........................> \n")D, ", class:", CGREEN + str(j + 1) + CEND, "for training")
    prediction_time = time.time()
    # batch = [batch, batch, batch, batch, batch]
    # y_pred_all = model['net'].predict([np.squeeze(batch[index]), np.squeeze(batch[index]), np.squeeze(batch[index]), np.squeeze(batch[index]), np.squeeze(batch[index])],
    #                               settings['batch_size'])
    y_pred_all = model['net'].predict(batch_array, settings['batch_size'])

    print("Prediction or loading learned model: ", round(time.time() - prediction_time), "sec")

    # for i in range(0, 5):
    # print("y_pred_all[", i, "].shape[0]", y_pred_all[i].shape[0])
    print("Prediction final stage for input:", '\x1b[6;30;41m' + str(index + 1) + '\x1b[0m', ", class:",
          CGREEN + str(w_class) + CEND, "\n")
    # y_pred = y_pred_all[index]
    # y_pred = y_pred_all
    if this_size == 5:
        y_pred = y_pred_all[index]
        [x, y, z] = np.stack(centers[index], axis=1)
        seg_image[x, y, z] = y_pred[:, 1]
        if settings['debug'] is True:
            print("...done!")
    else:
        y_pred = y_pred_all[index]
        [x, y, z] = np.stack(centers[index], axis=1)
        seg_image[x, y, z] = y_pred[:, 1]
        if settings['debug'] is True:
            print("...done!")


    print("Index to be saved is:", index)
    if save_nifti:
           out_scan = nib.Nifti1Image(seg_image, affine=flair_image.affine)
           out_scan.to_filename(os.path.join(settings['test_folder'],
                                           settings['run_testing'],
                                           settings['model_name'],
                                           settings['test_name']))


    return seg_image

def check_tolerance_error(input_scan, settings, voxel_size):
    """
    check that the output volume is higher than the minimum accuracy
    given by the
    parameter min_error
    """

    from scipy import ndimage

    t_bin = settings['t_bin']
    l_min = settings['l_min']

    # get voxel size in mm^3
    voxel_size = np.prod(voxel_size) / 1000.0

    # threshold input segmentation
    output_scan = np.zeros_like(input_scan)
    t_segmentation = input_scan >= t_bin

    # filter candidates by size and store those > l_min
    labels, num_labels = ndimage.label(t_segmentation)
    label_list = np.unique(labels)
    num_elements_by_lesion = ndimage.labeled_comprehension(t_segmentation,
                                                           labels,
                                                           label_list,
                                                           np.sum,
                                                           float, 0)

    for l in range(len(num_elements_by_lesion)):
        if num_elements_by_lesion[l] > l_min:
            # assign voxels to output
            current_voxels = np.stack(np.where(labels == l), axis=1)
            output_scan[current_voxels[:, 0],
                        current_voxels[:, 1],
                        current_voxels[:, 2]] = 1

    return (np.sum(output_scan == 1) * voxel_size) < settings['min_error']


def thresholded_voxels_from_learned_model(model, train_x_data, settings, index):

    scans = list(train_x_data.keys())


    seg_masks = []
    for scan, s in zip(list(train_x_data.keys()), list(range(len(scans)))):
        seg_mask = run_testing(model,
                             dict(list(train_x_data.items())[s:s+1]),
                             settings, index, save_nifti=False)
        seg_masks.append(seg_mask > 0.5)

        if settings['debug']:
            flair = nib.load(train_x_data[scan]['FLAIR'])
            tmp_seg = nib.Nifti1Image(seg_mask,
                                      affine=flair.affine)
            tmp_seg.to_filename(os.path.join(settings['weight_paths'],
                                             settings['model_name'],
                                             '.train',
                                             scan + '_it0.nii.gz'))

    # check candidate segmentations:
    # if no voxels have been selected, return candidate voxels on
    # FLAIR modality > 2
    flair_scans = [train_x_data[s]['FLAIR'] for s in scans]
    images = [load_nii(name).get_data() for name in flair_scans]
    images_norm = [normalize_data(im) for im in images]

    # seg_mask = [im > -2.5 if np.sum(seg) == 0 else seg
    #             for im, seg in zip(images_norm, seg_masks)]
    #
    #
    num = 1
    for seg in seg_masks:
        if not seg.any():
            print('\x1b[6;30;41m' + 'Warning:' + '\x1b[0m', 'after evaluating the training scan number:', num,' and applying probability higher than 0.5, no voxels have been selected, list contains empty element!' )
            print('')
            num = num + 1

    # seg_mask = [im > -2.5 if np.sum(seg) == 0 else seg
    #             for im, seg in zip(images_norm, seg_masks)]

    seg_mask = [im > -2.5 if not seg.any() else seg
                 for im, seg in zip(images_norm, seg_masks)]

    return seg_mask

def segmentation_final_process(input_scan,
                              settings,
                              save_nifti=True,
                              orientation=np.eye(4)):


    from scipy import ndimage

    t_bin = settings['t_bin']
    l_min = settings['l_min']
    output_scan = np.zeros_like(input_scan)

    # threshold input segmentation
    t_segmentation = input_scan >= t_bin

    # filter candidates by size and store those > l_min
    labels, num_labels = ndimage.label(t_segmentation)
    label_list = np.unique(labels)
    num_elements_by_lesion = ndimage.labeled_comprehension(t_segmentation,
                                                           labels,
                                                           label_list,
                                                           np.sum,
                                                           float, 0)

    for l in range(len(num_elements_by_lesion)):
        if num_elements_by_lesion[l] > l_min:
            # assign voxels to output
            current_voxels = np.stack(np.where(labels == l), axis=1)
            output_scan[current_voxels[:, 0],
                        current_voxels[:, 1],
                        current_voxels[:, 2]] = 1

    # save the output segmentation as Nifti1Image

    if save_nifti:
        nifti_out = nib.Nifti1Image(output_scan,
                                    affine=orientation)
        nifti_out.to_filename(os.path.join(settings['test_folder'],
                                           settings['run_testing'],
                                           settings['model_name'],
                                           settings['test_name']))

    return output_scan
