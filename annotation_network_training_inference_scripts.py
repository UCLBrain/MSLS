
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
import click
import shutil
import os
import sys
import platform
from timeit import time
import configparser
import numpy as np
from src.annotation_network_preprocess import preprocess
from src.annotation_network_postprocess import inverse_transformation
from src.annotation_network_load_settings import load_settings, print_settings
CURRENT_PATH = CURRENT_PATH = os.path.split(os.path.realpath(__file__))[0]
sys.path.append(os.path.join(CURRENT_PATH, 'libs'))


# check and remove the folder which dose not contain the necessary modalities before prepossessing step

def check_inputs(current_folder, settings, choice):
    """
    checking input errors, fixing  and writing it into the Input Issue Report File


    """
    erf =os.path.join(CURRENT_PATH, 'InputIssueReportfile.txt')
    f = open(erf, "a")

    if os.path.isdir(os.path.join(settings['train_folder'], current_folder)):
        if len(os.listdir(os.path.join(settings['train_folder'], current_folder))) == 0:
           print(('Directory:', current_folder, 'is empty'))
           print('Warning: if the  directory is not going to be removed, the Training could be later stopped!')
           if click.confirm('The empty directory will be removed. Do you want to continue?', default=True):
             f.write("The empty directory: %s has been removed from Training set!" % current_folder + os.linesep)
             f.close()
             shutil.rmtree(os.path.join(settings['train_folder'], current_folder), ignore_errors=True)
             return
           return
    else:
        pass

    if choice == 'training':
        modalities = settings['modalities'][:] + ['lesion']
        image_mods = settings['image_mods'][:] + settings['roi_mods'][:]
    else:
        modalities = settings['modalities'][:]
        image_mods = settings['image_mods'][:]

    if settings['debug']:
        print("> DEBUG:", "number of input sequences to find:", len(modalities))


    print("> PRE:", current_folder, "identifying input modalities")

    found_modalities = 0
    if os.path.isdir(os.path.join(settings['train_folder'], current_folder)):
        masks = [m for m in os.listdir(os.path.join(settings['train_folder'], current_folder)) if m.find('.nii') > 0]
        pass  # do your stuff here for directory
    else:
        # shutil.rmtree(os.path.join(settings['train_folder'], current_folder), ignore_errors=True)
        print(('The file:', current_folder, 'is not part of training'))
        print('Warning: if the  file is not going to be removed, the Training could be later stopped!')
        if click.confirm('The file will be removed. Do you want to continue?', default=True):
          f.write("The file: %s has been removed from Training set!" % current_folder + os.linesep)
          f.close()
          os.remove(os.path.join(settings['train_folder'], current_folder))
          return
        return




    for t, m in zip(image_mods, modalities):

        # check first the input modalities
        # find tag

        found_mod = [mask.find(t) if mask.find(t) >= 0
                     else np.Inf for mask in masks]

        if found_mod[np.argmin(found_mod)] is not np.Inf:
            found_modalities += 1


    # check that the minimum number of modalities are used
    if found_modalities < len(modalities):
           print("> ERROR:", current_folder, \
            "does not contain all valid input modalities")
           print('Warning: if the  folder is  not going to be removed, the Training could be later stopped!')
           if click.confirm('The folder will be removed. Do you want to continue?', default=True):
             f.write("The folder: %s has been removed from Training set!" % current_folder + os.linesep)
             f.close()
             shutil.rmtree(os.path.join(settings['train_folder'], current_folder), ignore_errors=True)


        #return True


def read_default_config():
    """
    Get the CNN_GUI configuration from file
    """
    default_config = configparser.SafeConfigParser()
    default_config.read(os.path.join(CURRENT_PATH, 'config', 'configuration.cfg'))


    # read user's configuration file
    settings = load_settings(default_config)
    settings['tmp_folder'] = CURRENT_PATH + '/tmp'
    settings['standard_lib'] = CURRENT_PATH + '/libs/standard'
    # set paths taking into account the host OS
    host_os = platform.system()
    if host_os == 'Linux' or 'Darwin':
        settings['niftyreg_path'] = CURRENT_PATH + '/libs/linux/niftyreg'
        settings['robex_path'] = CURRENT_PATH + '/libs/linux/ROBEX/runROBEX.sh'
        # settings['tensorboard_path'] = CURRENT_PATH + '/libs/bin/tensorboard'
        settings['test_slices'] = 256
    elif host_os == 'Windows':
        settings['niftyreg_path'] = os.path.normpath(
            os.path.join(CURRENT_PATH,
                         'libs',
                         'win',
                         'niftyreg'))

        settings['robex_path'] = os.path.normpath(
            os.path.join(CURRENT_PATH,
                         'libs',
                         'win',
                         'ROBEX',
                         'runROBEX.bat'))
        settings['test_slices'] = 256
    else:
        print("The OS system also here ...", host_os, "is not currently supported.")
        exit()

    # print settings when debugging
    if settings['debug']:
        print_settings(settings)

    return settings

def set_library(settings):

    device = str(settings['gpu_number'])
    print("DEBUG: ", device)
    os.environ['KERAS_BACKEND'] = 'tensorflow'
    os.environ["CUDA_VISIBLE_DEVICES"] = device




def trainingwork_annotation(settings):
    """
    Train the CNN_GUI network given the settings passed as parameter
    """
    CSELECTED = '\33[7m'
    CRED2    = '\33[91m'
    CEND      = '\33[0m'
    CBLINK2   = '\33[6m'
    # set GPU mode from the configuration file. Trying to update
    # the backend automatically from here in order to use either theano
    # or tensorflow backends
    from src.annotation_network_main_script import training_models
    from src.annotation_network_build_model import build_and_compile_models

    # define the training backend
    set_library(settings)

    # scan_list = os.listdir(settings['train_folder'])
    # scan_list.sort()
    # # check and remove the folder which dose not contain the necessary modalities before prepossessing step
    # for check in scan_list:
    #    check_inputs(check, settings, 'training')

    # update scan list after removing  the unnecessary folders before prepossessing step
    scan_list = os.listdir(settings['train_folder'])
    scan_list.sort()

    settings['task'] = 'training'
    settings['train_folder'] = os.path.normpath(settings['train_folder'])
    total_time = time.time()
    if settings['pre_processing'] is False:
        for scan in scan_list:



            # --------------------------------------------------
            # move things to a tmp folder before starting
            # --------------------------------------------------

            settings['tmp_scan'] = scan
            current_folder = os.path.join(settings['train_folder'], scan)
            settings['tmp_folder'] = os.path.normpath(os.path.join(current_folder,
                                                                   'tmp'))


            preprocess(current_folder, settings, CURRENT_PATH)



    # --------------------------------------------------
    # WM MS lesion training
    # - configure net and train
    # --------------------------------------------------
    if settings['pre_processing'] is False:
        default_config = configparser.ConfigParser()
        default_config.read(os.path.join(CURRENT_PATH , 'config', 'configuration.cfg'))
        default_config.set('completed', 'pre_processing', str(True))
        with open(os.path.join(CURRENT_PATH,
                               'config',
                               'configuration.cfg'), 'w') as configfile:
            default_config.write(configfile)


    seg_time = time.time()
    print(CSELECTED + "CNN_GUI: Starting training session"+ CEND)
    # select training scans
    train_x_data = {f: {m: os.path.join(settings['train_folder'], f, 'tmp', n)
                        for m, n in zip(settings['modalities'],
                                        settings['x_names'])}
                    for f in scan_list}

    print('selected training data:', train_x_data)

    # train_y_data = {f: {m: os.path.join(settings['train_folder'], f, 'tmp', n)
    #                     for m, n in zip(settings['labels'],
    #                                     settings['y_names'])}
    #                 for f in scan_list}

    print('train_y_label:', settings['all_isolated_label'])

    this_size = len(settings['all_isolated_label']) + 1

    train_y_data = {f: {m: os.path.join(settings['train_folder'], f, 'tmp', n)
                        for m, n in zip(range(1, this_size),
                                        settings['all_isolated_label'])}
                    for f in scan_list}

    print('selected lesion data:', train_y_data)

    settings['weight_paths'] = os.path.join(CURRENT_PATH, 'nets')
    settings['load_weights'] = False

    # train the model for the current scan

    print(CSELECTED + "CNN_GUI: training net with" + CEND, CRED2 + "%d" % (len(list(train_x_data.keys()))), "subjects"+ CEND  )

    # --------------------------------------------------
    # initialize the CNN_GUI and train the classifier
    # --------------------------------------------------
    model = build_and_compile_models(settings)

    if len(settings['all_isolated_label']) == 5:
        print('\x1b[6;30;44m' + 'CNN_GUI will train using binary masks          ' + '\x1b[0m')
    else:
        print('\x1b[6;30;44m' + 'CNN_GUI will train using non binary masks     ' + '\x1b[0m')

    model = training_models(model, train_x_data, train_y_data,  settings, CURRENT_PATH)

    print("> INFO: training time:", round(time.time() - seg_time), "sec")
    print("> INFO: total pipeline time: ", round(time.time() - total_time), "sec")

    print('\x1b[6;30;44m' + '...............................................' + '\x1b[0m')
    print('\x1b[6;30;44m' + 'First and second model are created successfully' + '\x1b[0m')
    print('\x1b[6;30;44m' + '...............................................' + '\x1b[0m')
    print('\x1b[6;30;41m' + 'Inference can be proceeded now!                ' + '\x1b[0m')

    # print("> INFO: All processes have been finished. Have a good day!")


def check_oututs(current_folder, settings, choice='testing'):
    """
    checking input errors, fixing  and writing it into the Input Issue Report File


    """
    erf =os.path.join(CURRENT_PATH, 'OutputIssueReportfile.txt')
    f = open(erf, "a")

    if os.path.isdir(os.path.join(settings['test_folder'], current_folder)):
        if len(os.listdir(os.path.join(settings['test_folder'], current_folder))) == 0:
           print(('Directory:', current_folder, 'is empty'))
           print('Warning: if the  directory is not going to be removed, the Testing could be later stopped!')
           if click.confirm('The empty directory will be removed. Do you want to continue?', default=True):
             f.write("The empty directory: %s has been removed from Testing set!" % current_folder + os.linesep)
             f.close()
             shutil.rmtree(os.path.join(settings['test_folder'], current_folder), ignore_errors=True)
             return
           return
    else:
        pass

    if choice == 'training':
        modalities = settings['modalities'][:] + ['lesion']
        image_mods = settings['image_mods'][:] + settings['roi_mods'][:]
    else:
        modalities = settings['modalities'][:]
        image_mods = settings['image_mods'][:]

    if settings['debug']:
        print("> DEBUG:", "number of input sequences to find:", len(modalities))


    print("> PRE:", current_folder, "identifying input modalities")

    found_modalities = 0
    if os.path.isdir(os.path.join(settings['test_folder'], current_folder)):
        masks = [m for m in os.listdir(os.path.join(settings['test_folder'], current_folder)) if m.find('.nii') > 0]
        pass  # do your stuff here for directory
    else:
        # shutil.rmtree(os.path.join(settings['train_folder'], current_folder), ignore_errors=True)
        print(('The file:', current_folder, 'is not part of testing'))
        print('Warning: if the  file is not going to be removed, the Testing could be later stopped!')
        if click.confirm('The file will be removed. Do you want to continue?', default=True):
          f.write("The file: %s has been removed from Testing set!" % current_folder + os.linesep)
          f.close()
          os.remove(os.path.join(settings['test_folder'], current_folder))
          return
        return




    for t, m in zip(image_mods, modalities):

        # check first the input modalities
        # find tag

        found_mod = [mask.find(t) if mask.find(t) >= 0
                     else np.Inf for mask in masks]

        if found_mod[np.argmin(found_mod)] is not np.Inf:
            found_modalities += 1


    # check that the minimum number of modalities are used
    if found_modalities < len(modalities):
           print("> ERROR:", current_folder, \
            "does not contain all valid input modalities")
           print('Warning: if the  folder is  not going to be removed, the Testing could be later stopped!')
           if click.confirm('The folder will be removed. Do you want to continue?', default=True):
             f.write("The folder: %s has been removed from Testing set!" % current_folder + os.linesep)
             f.close()
             shutil.rmtree(os.path.join(settings['test_folder'], current_folder), ignore_errors=True)





def inference_annotation(settings):
    """
    Infer segmentation given the input settings passed as parameters
    """

    # define the training backend
    set_library(settings)

    from src.annotation_network_main_script import testing_models
    from src.annotation_network_build_model import build_and_compile_models

    # --------------------------------------------------
    # net configuration
    # take into account if the pretrained models have to be used
    # all images share the same network model
    # --------------------------------------------------
    settings['full_train'] = True
    settings['load_weights'] = True
    settings['weight_paths'] = os.path.join(CURRENT_PATH, 'nets')
    settings['net_verbose'] = 0
    model = build_and_compile_models(settings)

    # --------------------------------------------------
    # process each of the scans
    # - image identification
    # - image registration
    # - skull-stripping
    # - WM segmentation
    # --------------------------------------------------

    scan_list = os.listdir(settings['test_folder'])
    scan_list.sort()
    # check and remove the folder which dose not contain the necessary modalities before prepossessing step
    for check in scan_list:
       check_oututs(check, settings)

    # update scan list after removing  the unnecessary folders before prepossessing step


    settings['task'] = 'testing'
    scan_list = os.listdir(settings['test_folder'])
    scan_list.sort()

    for scan in scan_list:

        total_time = time.time()
        settings['tmp_scan'] = scan
        # --------------------------------------------------
        # move things to a tmp folder before starting
        # --------------------------------------------------

        current_folder = os.path.join(settings['test_folder'], scan)
        settings['tmp_folder'] = os.path.normpath(
            os.path.join(current_folder,  'tmp'))

        # --------------------------------------------------
        # preprocess scans
        # --------------------------------------------------
        preprocess(current_folder, settings, CURRENT_PATH)

        # --------------------------------------------------
        # WM MS lesion inference
        # --------------------------------------------------
        seg_time = time.time()

        "> CNN_GUI:", scan, "running WM lesion segmentation"
        sys.stdout.flush()
        settings['run_testing'] = scan

        test_x_data = {scan: {m: os.path.join(settings['tmp_folder'], n)
                              for m, n in zip(settings['modalities'],
                                              settings['x_names'])}}

        this = testing_models(model, test_x_data, settings)
        
        if settings['register_modalities']:
             print("> INFO:", scan, "Inverting lesion segmentation masks")
             inverse_transformation(current_folder, settings)
            
        print("> INFO:", scan, "CNN_GUI Segmentation time: ", round(time.time() - seg_time), "sec")
        print("> INFO:", scan, "total pipeline time: ", round(time.time() - total_time), "sec")

        # remove tmps if not set
        if settings['save_tmp'] is False:
            try:
                os.rmdir(settings['tmp_folder'])
                os.rmdir(os.path.join(settings['current_folder'],
                                      settings['model_name']))
            except:
                pass

    print('\x1b[6;30;41m' + 'Inference has been proceeded' + '\x1b[0m')
    
