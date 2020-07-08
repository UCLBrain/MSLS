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
CEND      = '\33[0m'
CBOLD     = '\33[1m'
CITALIC   = '\33[3m'
CURL      = '\33[4m'
CBLINK    = '\33[5m'
CBLINK2   = '\33[6m'
CSELECTED = '\33[7m'

CBLACK  = '\33[30m'
CRED    = '\33[31m'
CGREEN  = '\33[32m'
CYELLOW = '\33[33m'
CBLUE   = '\33[34m'
CVIOLET = '\33[35m'
CBEIGE  = '\33[36m'
CWHITE  = '\33[37m'

CBLACKBG  = '\33[40m'
CREDBG    = '\33[41m'
CGREENBG  = '\33[42m'
CYELLOWBG = '\33[43m'
CBLUEBG   = '\33[44m'
CVIOLETBG = '\33[45m'
CBEIGEBG  = '\33[46m'
CWHITEBG  = '\33[47m'

CGREY    = '\33[90m'
CRED2    = '\33[91m'
CGREEN2  = '\33[92m'
CYELLOW2 = '\33[93m'
CBLUE2   = '\33[94m'
CVIOLET2 = '\33[95m'
CBEIGE2  = '\33[96m'
CWHITE2  = '\33[97m'

CGREYBG    = '\33[100m'
CREDBG2    = '\33[101m'
CGREENBG2  = '\33[102m'
CYELLOWBG2 = '\33[103m'
CBLUEBG2   = '\33[104m'
CVIOLETBG2 = '\33[105m'
CBEIGEBG2  = '\33[106m'
CWHITEBG2  = '\33[107m'

def load_settings(default_config):

    settings = {}

    # model_name name (where trained weights are)
    settings['model_name'] = default_config.get('TrainTestSet', 'name')
    settings['train_folder'] = default_config.get('TrainTestSet', 'train_folder')
    settings['test_folder'] = default_config.get('TrainTestSet', 'inference_folder')
    settings['output_folder'] = '/output'
    settings['current_scan'] = 'scan'
    # settings['t1_name'] = default_config.get('TrainTestSet', 't1_name')
    # settings['flair_name'] = default_config.get('TrainTestSet', 'flair_name')
    settings['flair_mods'] = [el.strip() for el in
                             default_config.get('TrainTestSet',
                                                'flair_mods').split(',')]
    settings['t1_mods'] = [el.strip() for el in
                          default_config.get('TrainTestSet',
                                             't1_mods').split(',')]
    settings['mod3_mods'] = [el.strip() for el in
                            default_config.get('TrainTestSet',
                                               'mod3_mods').split(',')]
    settings['mod4_mods'] = [el.strip() for el in
                            default_config.get('TrainTestSet',
                                               'mod4_mods').split(',')]
    settings['roi_mods'] = [el.strip() for el in
                           default_config.get('TrainTestSet',
                                              'roi_mods').split(',')]


    settings['l1_mods'] = [el.strip() for el in
                             default_config.get('TrainTestSet',
                                                'l1_mods').split(',')]
    settings['l2_mods'] = [el.strip() for el in
                             default_config.get('TrainTestSet',
                                                'l2_mods').split(',')]
    settings['l3_mods'] = [el.strip() for el in
                             default_config.get('TrainTestSet',
                                                'l3_mods').split(',')]
    settings['l4_mods'] = [el.strip() for el in
                             default_config.get('TrainTestSet',
                                                'l4_mods').split(',')]
    settings['l5_mods'] = [el.strip() for el in
                             default_config.get('TrainTestSet',
                                                'l5_mods').split(',')]

    settings['all_isolated_label'] = [el.strip() for el in
                           default_config.get('TrainTestSet',
                                              'all_isolated_label').split(',')]




    # settings['ROI_name'] = default_config.get('TrainTestSet', 'ROI_name')
    settings['debug'] = default_config.get('TrainTestSet', 'debug')

    modalities = [str(settings['flair_mods'][0]),
                  settings['t1_mods'][0],
                  settings['mod3_mods'][0],
                  settings['mod4_mods'][0]]
    names = ['FLAIR', 'T1', 'MOD3', 'MOD4']

    labels = [settings['l1_mods'][0],
              settings['l2_mods'][0],
              settings['l3_mods'][0],
              settings['l4_mods'][0],
              settings['l5_mods'][0]]

    label_names = ['LB1', 'LB2', 'LB3', 'LB4', 'LB5']

    settings['labels'] = [n for n, m in
                             zip(label_names, labels) if m != 'None']

    settings['y_names'] = [n + '.nii.gz' for n, m in
                          zip(label_names, labels) if m != 'None']

    settings['modalities'] = [n for n, m in
                             zip(names, modalities) if m != 'None']
    settings['label_mods'] = [m for m in labels if m != 'None']
    settings['image_mods'] = [m for m in modalities if m != 'None']
    settings['x_names'] = [n + '_brain.nii.gz' for n, m in
                          zip(names, modalities) if m != 'None']

    settings['out_name'] = 'out_seg.nii.gz'

    # preprocessing
    settings['register_modalities'] = (default_config.get('TrainTestSet',
                                                         'register_modalities'))

    # settings['register_modalities_kind'] = (default_config.get('TrainTestSet',
    #                                                      'register_modalities_Kind'))

    settings['reg_space'] = (default_config.get('TrainTestSet',
                                                         'reg_space'))                                                     


    settings['denoise'] = (default_config.get('TrainTestSet',
                                             'denoise'))
    settings['denoise_iter'] = (default_config.getint('TrainTestSet',
                                                     'denoise_iter'))
    settings['bias_iter'] = (default_config.getint('TrainTestSet',
                                                     'bias_iter'))

    settings['number_of_classes'] = (default_config.getint('TrainTestSet',
                                                  'number_of_classes'))
    settings['bias_smooth'] = (default_config.getint('TrainTestSet',
                                                     'bias_smooth')) 
    settings['bias_type'] = (default_config.getint('TrainTestSet',
                                                     'bias_type'))                                                   
    settings['bias_choice'] = (default_config.get('TrainTestSet',
                                                     'bias_choice'))



    settings['bias_correction'] = (default_config.get('TrainTestSet',
                                                     'bias_correction'))

    settings['batch_prediction'] = (default_config.get('TrainTestSet',
                                                     'batch_prediction'))


    settings['mnist_dataset'] = (default_config.get('TrainTestSet',
                                                     'mnist_dataset'))


    settings['ms_dataset'] = (default_config.get('TrainTestSet',
                                                     'ms_dataset'))

    settings['annotation_network'] = (default_config.get('TrainTestSet',
                                                     'annotation_network'))
    settings['segmentation_network'] = (default_config.get('TrainTestSet',
                                                     'segmentation_network'))


    settings['skull_stripping'] = (default_config.get('TrainTestSet',
                                                     'skull_strippingping'))
    settings['save_tmp'] = (default_config.get('TrainTestSet', 'save_tmp'))

    # net settings
    # settings['gpu_mode'] = default_config.get('model', 'gpu_mode')
    settings['gpu_number'] = default_config.getint('TrainTestSet', 'gpu_number')
    settings['pretrained'] = default_config.get('TrainTestSet', 'pretrained')
    settings['min_th'] = 0.5
    settings['fully_convolutional'] = False
    settings['patch_size'] = (11, 11, 11)
    settings['weight_paths'] = None
    settings['train_split'] = default_config.getfloat('TrainTestSet', 'train_split')
    settings['max_epochs'] = default_config.getint('TrainTestSet', 'max_epochs')
    settings['patience'] = default_config.getint('TrainTestSet', 'patience')
    settings['batch_size'] = default_config.getint('TrainTestSet', 'batch_size')
    settings['net_verbose'] = default_config.getint('TrainTestSet', 'net_verbose')

    settings['tensorboard'] = default_config.get('tensorboard', 'tensorboard_folder')
    settings['port'] = default_config.getint('tensorboard', 'port')
    


    # settings['load_weights'] = default_config.get('model', 'load_weights')
    settings['load_weights'] = True
    settings['randomize_train'] = True

    # post processing settings
    settings['t_bin'] = default_config.getfloat('TrainTestSet', 't_bin')
    settings['l_min'] = default_config.getint('TrainTestSet', 'l_min')
    settings['min_error'] = default_config.getfloat('TrainTestSet',
                                                   'min_error')

    # training settings  model_1_train
    settings['full_train'] = (default_config.get('TrainTestSet', 'full_train'))
    settings['model_1_train'] = (default_config.get('completed', 'model_1_train'))
    settings['model_2_train'] = (default_config.get('completed', 'model_2_train'))
    settings['pre_processing'] = (default_config.get('completed', 'pre_processing'))
    settings['pretrained_model'] = default_config.get('TrainTestSet',
                                                     'pretrained_model')

    settings['balanced_training'] = default_config.get('TrainTestSet',
                                                      'balanced_training')

    settings['fract_negative_positive'] = default_config.getfloat('TrainTestSet',
                                                                 'fraction_negatives')


    settings['fract_negative_positive_CV'] = default_config.getfloat('TrainTestSet',
                                                                 'fraction_negatives_CV')
    settings['num_layers'] = None

    settings = parse_values_to_types(settings)
    return settings


def parse_values_to_types(settings):
    """
    process values into types
    """

    keys = list(settings.keys())
    for k in keys:
        value = settings[k]
        if value == 'True':
            settings[k] = True
        if value == 'False':
            settings[k] = False

    return settings


def print_settings(settings):
    print('\x1b[6;30;45m' + '                   ' + '\x1b[0m')
    print('\x1b[6;30;45m' + 'Train/Test settings' + '\x1b[0m')
    print('\x1b[6;30;45m' + '                   ' + '\x1b[0m')
    print(" ")
    keys = list(settings.keys())
    for k in keys:
        print(CRED + k, ':' + CEND, settings[k])
    print('\x1b[6;30;45m' + '                   ' + '\x1b[0m')
