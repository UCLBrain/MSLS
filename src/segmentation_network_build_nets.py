
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




from Mkeras.layers import Dense, Dropout, Flatten, Input
from Mkeras.layers.convolutional import Conv3D, MaxPooling3D
from Mkeras.layers.advanced_activations import PReLU as prelu
from Mkeras.layers.normalization import BatchNormalization as BN
from Mkeras import backend as K
from Mkeras import regularizers
from Mkeras.models import Model
import Mkeras
# K.set_image_dim_ordering('th')
K.set_image_data_format('channels_first')

def build_network(settings, model_name=None):



    # print("adaptive deep learning network architecture")

    print('\x1b[6;30;41m' + '                                                            ' + '\x1b[0m')
    print('\x1b[6;30;41m' + 'Multi class Deep Learning Network With Adaptive Architecture' + '\x1b[0m')

    if model_name == 'First':
        print('\x1b[6;30;41m' + '                 (' + model_name + ' Model)                              ' + '\x1b[0m')
    else:
        print('\x1b[6;30;41m' + '                 (' + model_name + ' Model)                             ' + '\x1b[0m')
    print('\x1b[6;30;41m' + '                                                            ' + '\x1b[0m')


    channels = len(settings['modalities'])

    this_size = len(settings['all_isolated_label'])

    # adaptive deep learning network architecture
    net_IN = []
    for i in range(0, this_size):
        this_name = 'input' + str(i + 1)
        net_IN.append(Input(name=this_name, shape=(channels,) + settings['patch_size']))
        # net_IN[i] = Input(name=this_name, shape=(channels,) + settings['patch_size'])
    # if label will be a binary label
    if this_size > 5:
        merged = Mkeras.layers.Concatenate(axis=1)(net_IN)
    # net_input = Input(name='in1', shape=(channels,) + settings['patch_size'])
        layer = Conv3D(filters=32, kernel_size=(3, 3, 3),
                   name='conv1_1',
                   activation=None,
                   padding="same")(merged)
    else:
        net_input1 = Input(name='in1', shape=(channels,) + settings['patch_size'])
        net_input2 = Input(name='in2', shape=(channels,) + settings['patch_size'])
        net_input3 = Input(name='in3', shape=(channels,) + settings['patch_size'])
        net_input4 = Input(name='in4', shape=(channels,) + settings['patch_size'])
        net_input5 = Input(name='in5', shape=(channels,) + settings['patch_size'])


        merged = Mkeras.layers.Concatenate(axis=1)([net_input1,net_input2,net_input3,net_input4,net_input5])

        # net_input = [net_input1,net_input2,net_input3,net_input4,net_input5]
        layer = Conv3D(filters=32, kernel_size=(3, 3, 3),
                       name='conv1_1',
                       activation=None,
                       padding="same")(merged)



    layer = BN(name='bn_1_1', axis=1)(layer)
    layer = prelu(name='prelu_conv1_1')(layer)
    layer = Conv3D(filters=32,
                   kernel_size=(3, 3, 3),
                   name='conv1_2',
                   activation=None,
                   padding="same")(layer)
    layer = BN(name='bn_1_2', axis=1)(layer)
    layer = prelu(name='prelu_conv1_2')(layer)
    layer = MaxPooling3D(pool_size=(2, 2, 2),
                         strides=(2, 2, 2))(layer)
    layer = Conv3D(filters=64,
                   kernel_size=(3, 3, 3),
                   name='conv2_1',
                   activation=None,
                   padding="same")(layer)
    layer = BN(name='bn_2_1', axis=1)(layer)
    layer = prelu(name='prelu_conv2_1')(layer)
    layer = Conv3D(filters=64,
                   kernel_size=(3, 3, 3),
                   name='conv2_2',
                   activation=None,
                   padding="same")(layer)
    layer = BN(name='bn_2_2', axis=1)(layer)
    layer = prelu(name='prelu_conv2_2')(layer)
    layer = MaxPooling3D(pool_size=(2, 2, 2),
                         strides=(2, 2, 2))(layer)
    layer = Flatten()(layer)
    layer = Dropout(name='dr_d1', rate=0.5)(layer)
    layer = Dense(units=256,  activation=None, name='d1')(layer)
    layer = prelu(name='prelu_d1')(layer)
    layer = Dropout(name='dr_d2', rate=0.5)(layer)
    layer = Dense(units=128,  activation=None, name='d2')(layer)
    layer = prelu(name='prelu_d2')(layer)
    layer = Dropout(name='dr_d3', rate=0.5)(layer)
    layer = Dense(units=64,  activation=None, name='d3')(layer)
    layer = prelu(name='prelu_d3')(layer)
    net_OUT = []
    for i in range(0, int(this_size / 5)):
        this_name = 'output' + str(i + 1)
        net_OUT.append(Dense(units=2, name=this_name, activation='softmax')(layer))

    if this_size > 5:
         model = Model(inputs=net_IN, outputs=net_OUT)


    else:
         net_output = Dense(units=2, name='approximated_5to1_output', activation='softmax')(layer)
         model = Model(inputs=[net_input1, net_input2, net_input3, net_input4, net_input5], outputs=net_output)

    print('')
    if model_name == 'First':
        if this_size == 5:
            print('\x1b[6;30;43m' + model_name + ' Model Architecture:                ' + '\x1b[0m')
            print('\x1b[6;30;43m' + '3D Convolutional Neural Network          ' + '\x1b[0m')
            print('\x1b[6;30;43m' + str(this_size) + ' Input tensor(s) and ' + str(
                int(this_size / 5)) + ' Output tensor(s)  ' + '\x1b[0m')
        else:
            print('\x1b[6;30;43m' + model_name + ' Model Architecture:                  ' + '\x1b[0m')
            print('\x1b[6;30;43m' + '3D Convolutional Neural Network            ' + '\x1b[0m')
            print('\x1b[6;30;43m' + str(this_size) + ' Input tensor(s) and ' + str(
                int(this_size / 5)) + ' Output tensor(s)  ' + '\x1b[0m')

    else:
        if this_size == 5:
            print('\x1b[6;30;43m' + model_name + ' Model Architecture:               ' + '\x1b[0m')
            print('\x1b[6;30;43m' + '3D Convolutional Neural Network          ' + '\x1b[0m')
            print('\x1b[6;30;43m' + str(this_size) + ' Input tensor(s) and ' + str(
                int(this_size / 5)) + ' Output tensor(s)  ' + '\x1b[0m')
        else:
            print('\x1b[6;30;43m' + model_name + ' Model Architecture:                 ' + '\x1b[0m')
            print('\x1b[6;30;43m' + '3D Convolutional Neural Network            ' + '\x1b[0m')
            print('\x1b[6;30;43m' + str(this_size) + ' Input tensor(s) and ' + str(
                int(this_size / 5)) + ' Output tensor(s)  ' + '\x1b[0m')

    print('')


    return model


