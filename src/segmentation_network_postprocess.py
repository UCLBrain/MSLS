
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
import subprocess
import time
import platform
import sys

def  inverse_transformation(current_folder, settings):

    os_host = platform.system()
    if os_host == 'Windows':
        reg_transform = 'reg_transform.exe'
        reg_resample = 'reg_resample.exe'
    elif os_host == 'Linux' or 'Darwin':
        reg_transform = 'reg_transform'
        reg_resample = 'reg_resample'
    else:
        print("> ERROR: The OS system", os_host, "is not currently supported.")


    if os_host == 'Windows':
        reg_transform_path = os.path.join(settings['niftyreg_path'], reg_transform)
        reg_resample_path = os.path.join(settings['niftyreg_path'], reg_resample)
    elif os_host == 'Linux':
        reg_transform_path = os.path.join(settings['niftyreg_path'], reg_transform)
        reg_resample_path = os.path.join(settings['niftyreg_path'], reg_resample)
    elif os_host == 'Darwin':
        reg_transform_path = reg_transform
        reg_resample_path = reg_resample
    else:
        print('Please install first  NiftyReg in your mac system and try again!')
        sys.stdout.flush()
        time.sleep(1)
        os.kill(os.getpid(), signal.SIGTERM)
    print('running ....> ', reg_transform_path)
    print('running ....> ', reg_resample_path)


    # inverse transformation


    if  settings['reg_space'] != 'FlairtoT1' and  settings['reg_space'] != 'T1toFlair':
        try:
            subprocess.check_output([reg_transform_path, '-invAff',
                                     os.path.join(settings['tmp_folder'],
                                                  'FLAIR_transf.txt'),
                                     os.path.join(settings['tmp_folder'],
                                                  'inv_FLAIR_transf.txt')])
        except:
            print("> ERROR: computing the inverse transformation matrix.\
             Quitting program.")
            time.sleep(1)
            os.kill(os.getpid(), signal.SIGTERM)

        print("> POST: registering output segmentation masks back to FLAIR")

        current_model_name = os.path.join(current_folder, settings['model_name'])
        list_scans = os.listdir(current_model_name)

        for file in list_scans:

            # compute the inverse transformation
            current_name = file[0:file.find('.')]
            try:
                subprocess.check_output([reg_resample_path,
                                         '-ref', os.path.join(settings['tmp_folder'],
                                                              'FLAIR.nii.gz'),
                                         '-flo', os.path.join(current_model_name,
                                                              file),
                                         '-trans', os.path.join(settings['tmp_folder'],
                                                                'inv_FLAIR_transf.txt'),
                                         '-res', os.path.join(current_model_name,
                                                              current_name + '_FLAIR.nii.gz'),
                                         '-inter', '0'])
            except:
                print("> ERROR: resampling ", current_name, "Quitting program.")
                time.sleep(1)
                os.kill(os.getpid(), signal.SIGTERM)

    if settings['reg_space'] == 'FlairtoT1':
        try:
            subprocess.check_output([reg_transform_path, '-invAff',
                                 os.path.join(settings['tmp_folder'],
                                              'FLAIR_transf.txt'),
                                 os.path.join(settings['tmp_folder'],
                                              'inv_FLAIR_transf.txt')])
        except:
            print("> ERROR: computing the inverse transformation matrix.\
            Quitting program.")
            time.sleep(1)
            os.kill(os.getpid(), signal.SIGTERM)

        print("> POST: registering output segmentation masks back to FLAIR")

        current_model_name = os.path.join(current_folder, settings['model_name'])
        list_scans = os.listdir(current_model_name)

        for file in list_scans:

          # compute the inverse transformation
            current_name = file[0:file.find('.')]
            try:
                subprocess.check_output([reg_resample_path,
                                     '-ref', os.path.join(settings['tmp_folder'],
                                                          'FLAIR.nii.gz'),
                                     '-flo', os.path.join(current_model_name,
                                                          file),
                                     '-trans', os.path.join(settings['tmp_folder'],
                                                            'inv_FLAIR_transf.txt'),
                                     '-res', os.path.join(current_model_name,
                                                          current_name + '_FLAIR.nii.gz'),
                                     '-inter', '0'])
            except:
                print("> ERROR: resampling ",  current_name, "Quitting program.")
                time.sleep(1)
                os.kill(os.getpid(), signal.SIGTERM)

    if settings['reg_space'] == 'T1toFlair':
        try:
            subprocess.check_output([reg_transform_path, '-invAff',
                                     os.path.join(settings['tmp_folder'],
                                                  'T1_transf.txt'),
                                     os.path.join(settings['tmp_folder'],
                                                  'inv_T1_transf.txt')])
        except:
            print("> ERROR: computing the inverse transformation matrix.\
             Quitting program.")
            time.sleep(1)
            os.kill(os.getpid(), signal.SIGTERM)

        print("> POST: registering output segmentation masks back to T1")

        current_model_name = os.path.join(current_folder, settings['model_name'])
        list_scans = os.listdir(current_model_name)

        for file in list_scans:

            # compute the inverse transformation
            current_name = file[0:file.find('.')]
            try:
                subprocess.check_output([reg_resample_path,
                                         '-ref', os.path.join(settings['tmp_folder'],
                                                              'T1.nii.gz'),
                                         '-flo', os.path.join(current_model_name,
                                                              file),
                                         '-trans', os.path.join(settings['tmp_folder'],
                                                                'inv_T1_transf.txt'),
                                         '-res', os.path.join(current_model_name,
                                                              current_name + '_T1.nii.gz'),
                                         '-inter', '0'])
            except:
                print("> ERROR: resampling ", current_name, "Quitting program.")
                time.sleep(1)
                os.kill(os.getpid(), signal.SIGTERM)
