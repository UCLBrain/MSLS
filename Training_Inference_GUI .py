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

import configparser
import argparse
import platform
import subprocess
import os
import signal
import queue
import threading
from __init__ import __version__
from tkinter import Frame, LabelFrame, Label, END, Tk
from tkinter import Entry, Button, Checkbutton, OptionMenu, Toplevel
from tkinter import BooleanVar, StringVar, IntVar, DoubleVar
from tkinter.filedialog import askdirectory
from tkinter.ttk import Notebook
from PIL import Image, ImageTk
import webbrowser
from segmentation_network_training_inference_scripts import trainingwork, inference, read_default_config
from annotation_network_training_inference_scripts import trainingwork_annotation, inference_annotation, read_default_config

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

class Animated_GIF(Label, object):
    def __init__(self, master, path, forever=True):
        self._master = master
        self._loc = 0
        self._forever = forever

        self._is_running = False

        im = Image.open(path)
        self._frames = []
        i = 0
        try:
            while True:
                # photoframe = ImageTk.PhotoImage(im.copy().convert('RGBA'))
                photoframe = ImageTk.PhotoImage(im)
                self._frames.append(photoframe)

                i += 1
                im.seek(i)
        except EOFError:
            pass

        self._last_index = len(self._frames) - 1

        try:
            self._delay = im.info['duration']
        except:
            self._delay = 1000

        self._callback_id = None

        super(Animated_GIF, self).__init__(master, image=self._frames[0])

    def start_animation(self, frame=None):
        if self._is_running: return

        if frame is not None:
            self._loc = 0
            self.configure(image=self._frames[frame])

        self._master.after(self._delay, self._animate_GIF)
        self._is_running = True

    def stop_animation(self):
        if not self._is_running: return

        if self._callback_id is not None:
            self.after_cancel(self._callback_id)
            self._callback_id = None

        self._is_running = False

    def _animate_GIF(self):
        self._loc += 1
        self.configure(image=self._frames[self._loc])

        if self._loc == self._last_index:
            if self._forever:
                self._loc = 0
                self._callback_id = self._master.after(self._delay, self._animate_GIF)
            else:
                self._callback_id = None
                self._is_running = False
        else:
            self._callback_id = self._master.after(self._delay, self._animate_GIF)

    def pack(self, start_animation=True, **kwargs):
        if start_animation:
            self.start_animation()

        super(Animated_GIF, self).pack(**kwargs)

    def grid(self, start_animation=True, **kwargs):
        if start_animation:
            self.start_animation()

        super(Animated_GIF, self).grid(**kwargs)

    def place(self, start_animation=True, **kwargs):
        if start_animation:
            self.start_animation()

        super(Animated_GIF, self).place(**kwargs)

    def pack_forget(self, **kwargs):
        self.stop_animation()

        super(Animated_GIF, self).pack_forget(**kwargs)

    def grid_forget(self, **kwargs):
        self.stop_animation()

        super(Animated_GIF, self).grid_forget(**kwargs)

    def place_forget(self, **kwargs):
        self.stop_animation()

        super(Animated_GIF, self).place_forget(**kwargs)




class CNN_GUI:

    def __init__(self, master, container):

        self.master = master
        master.title("Multi-Label Multi/Single-Class Image Segmentation, UCL 2020")

        # running on a container
        self.container = container

        # gui attributes
        self.path = os.getcwd()
        self.default_config = None
        self.user_new_settings = None
        self.current_folder = os.getcwd()
        self.list_train_pretrained_nets = []
        self.setting_bias_choice = StringVar() 
        
        self.list_bias = ['All', 'FLAIR', 'T1', 'MOD3', 'MOD4']
        
        self.list_standard_space_reg_list = ['FlairtoT1', 'T1toFlair', 'avg152T1.nii.gz', 'avg152T1_brain.nii.gz',
        'FMRIB58_FA_1mm.nii.gz', 'FMRIB58_FA-skeleton_1mm.nii.gz', 'Fornix_FMRIB_FA1mm.nii.gz', 'LowerCingulum_1mm.nii.gz',
        'MNI152lin_T1_1mm.nii.gz','MNI152lin_T1_1mm_brain.nii.gz', 'MNI152lin_T1_1mm_subbr_mask.nii.gz',
        'MNI152lin_T1_2mm.nii.gz', 'MNI152lin_T1_2mm_brain.nii.gz', 'MNI152lin_T1_2mm_brain_mask.nii.gz',
        'MNI152_T1_0.5mm.nii.gz', 'MNI152_T1_1mm.nii.gz', 'MNI152_T1_1mm_brain.nii.gz',
        'MNI152_T1_1mm_brain_mask.nii.gz', 'MNI152_T1_1mm_brain_mask_dil.nii.gz', 'MNI152_T1_1mm_first_brain_mask.nii.gz',
        'MNI152_T1_1mm_Hipp_mask_dil8.nii.gz', 'MNI152_T1_2mm.nii.gz', 'MNI152_T1_2mm_b0.nii.gz', 'MNI152_T1_2mm_brain.nii.gz',
        'MNI152_T1_2mm_brain_mask.nii.gz', 'MNI152_T1_2mm_brain_mask_deweight_eyes.nii.gz', 'MNI152_T1_2mm_brain_mask_dil.nii.gz',
        'MNI152_T1_2mm_brain_mask_dil1.nii.gz', 'MNI152_T1_2mm_edges.nii.gz', 'MNI152_T1_2mm_eye_mask.nii.gz',
        'MNI152_T1_2mm_LR-masked.nii.gz', 'MNI152_T1_2mm_skull.nii.gz', 'MNI152_T1_2mm_strucseg.nii.gz',        
        'MNI152_T1_2mm_strucseg_periph.nii.gz', 'MNI152_T1_2mm_VentricleMask.nii.gz']
        self.list_test_nets = []
        self.version = __version__
        # if self.container is False:
            # version_number
          #  self.commit_version = subprocess.check_output(
           #     ['git', 'rev-parse', 'HEAD'])

        # queue and thread parameters. All processes are embedded
        # inside threads to avoid freezing the application
        self.train_task = None
        self.test_task = None
        self.test_queue = queue.Queue()
        self.train_queue = queue.Queue()

        # --------------------------------------------------
        # parameters. Mostly from the config/*.cfg files
        # --------------------------------------------------

        # data parameters
        self.setting_training_folder = StringVar()
        self.all_label = StringVar()
        self.setting_test_folder = StringVar()
        self.setting_tensorboard_folder = StringVar()
        self.setting_port_value = IntVar()
        self.setting_FLAIR_mod = StringVar()
        self.setting_label1_mod = StringVar()
        self.setting_label2_mod = StringVar()
        self.setting_label3_mod = StringVar()
        self.setting_label4_mod = StringVar()
        self.setting_label5_mod = StringVar()
        self.setting_PORT_mod = IntVar()
        self.setting_T1_mod = StringVar()
        self.setting_MOD3_mod = StringVar()
        self.setting_MOD4_mod = StringVar()
        self.setting_mask_mod = StringVar()
        self.setting_model_mod = StringVar()
        self.setting_register_modalities = BooleanVar()
        self.setting_bias_correction = BooleanVar()
        self.setting_batch_prediction = BooleanVar()
        self.setting_Mnist_dataset = BooleanVar()
        self.setting_MS_dataset = BooleanVar()
        self.setting_Annotation = BooleanVar()
        self.setting_Segmentation = BooleanVar()
        self.setting_bin = BooleanVar()
        self.setting_multi = BooleanVar()

        self.setting_register_modalities_Kind = StringVar()
        self.setting_skull_strippingping = BooleanVar()
        self.setting_denoise = BooleanVar()
        self.setting_denoise_iter = IntVar()
        self.setting_save_tmp = BooleanVar()
        self.setting_debug = BooleanVar()

        self.model_1_train = BooleanVar()
        self.model_2_train = BooleanVar()
        self.pre_processing = BooleanVar()


        # train parameters
        self.setting_net_folder = os.path.join(self.current_folder, 'nets')
        self.setting_use_pretrained_model = BooleanVar()
        self.setting_reg_space = StringVar()
        self.setting_pretrained_model = StringVar()
        self.setting_inference_model = StringVar()
        self.setting_num_layers = IntVar()
        self.setting_net_name = StringVar()
        self.setting_net_name.set('None')
        self.setting_balanced_dataset = StringVar()
        self.setting_fract_negatives = DoubleVar()
        self.setting_fract_negatives_cv = DoubleVar()
        # niter=10;
        # smooth=20;
        # betfparam=0.1;
        # type=1  # For FAST: 1 = T1w, 2 = T2w, 3 = PD
        self.setting_Bias_cor_niter = IntVar()
        self.setting_Bias_cor_smooth = IntVar()
        self.setting_Bias_cor_type = IntVar()

        self.setting_Number_of_classes = IntVar()
        # model parameters
        self.setting_predefiend_reg1 = None
        self.setting_predefiend_reg2 = 'T1toFlair'
        self.setting_pretrained = None
        self.setting_min_th = DoubleVar()
        self.setting_patch_size = IntVar()
        self.setting_weight_paths = StringVar()
        self.setting_load_weights = BooleanVar()
        self.setting_train_split = DoubleVar()
        self.setting_max_epochs = IntVar()
        self.setting_patience = IntVar()
        self.setting_batch_size = IntVar()
        self.setting_net_verbose = IntVar()
        self.setting_t_bin = DoubleVar()
        self.setting_l_min = IntVar()
        self.setting_min_error = DoubleVar()
        self.setting_mode = BooleanVar()
        self.setting_gpu_number = IntVar()

        # load the default configuration from the conf file
        self.read_default_configuration()

        # self frame (tabbed notebook)
        self.note = Notebook(self.master)
        self.note.pack()

        os.system('cls' if platform.system() == 'Windows' else 'clear')
        # image = Image.open('U1.jpg')
        # image.show()
        print("##################################################")
        print('\x1b[6;30;45m' + 'Multi/Single-Class Lesion Segmentation    ' + '\x1b[0m')
        print('\x1b[6;30;42m' + 'Medical Physics and Biomedical Engineering' + '\x1b[0m')
        print('\x1b[6;30;44m' + 'UCL - 2020                                ' + '\x1b[0m')
        print('\x1b[6;30;41m' + 'Kevin Bronik and Le Zhang                 ' + '\x1b[0m')
        print("##################################################")

        # --------------------------------------------------
        # training tab
        # --------------------------------------------------
        self.train_frame = Frame()
        self.note.add(self.train_frame, text="Training")
        self.test_frame = Frame()
        self.note.add(self.test_frame, text="Inference")

        # label frames
        cl_s = 6
        self.tr_frame = LabelFrame(self.train_frame, text="Training images:")
        self.tr_frame.grid(row=0, columnspan=cl_s, sticky='WE',
                           padx=5, pady=5, ipadx=5, ipady=5)
        self.model_frame = LabelFrame(self.train_frame, text="Deep Neural Network‎ model:")
        self.model_frame.grid(row=5, columnspan=cl_s, sticky='WE',
                              padx=5, pady=5, ipadx=5, ipady=5)
        self.tb_frame = LabelFrame(self.train_frame, text="TensorBoard Option:")
        self.tb_frame.grid(row=6, columnspan=cl_s, sticky='WE',
                              padx=5, pady=5, ipadx=5, ipady=5)

        # training settings
        self.inFolderLbl = Label(self.tr_frame, text="Training folder:")
        self.inFolderLbl.grid(row=0, column=0, sticky='E', padx=5, pady=2)
        self.inFolderTxt = Entry(self.tr_frame)
        self.inFolderTxt.grid(row=0,
                              column=1,
                              columnspan=5,
                              sticky="W",
                              pady=3)
        self.inFileBtn = Button(self.tr_frame, text="Browse ...",
                                command=self.training_path)
        self.inFileBtn.grid(row=0,
                            column=2,
                            columnspan=1,
                            sticky='W',
                            padx=5,
                            pady=1)

        self.settingsBtn = Button(self.tr_frame,
                                 text="Setting",
                                 command=self.Add_Settings)
        self.settingsBtn.grid(row=0,
                             column=3,
                             columnspan=1,
                             sticky="W",
                             padx=(95, 1),
                             pady=1)

        self.TensorBoard_inFolderLbl = Label(self.tb_frame , text="TensorBoard folder:")
        self.TensorBoard_inFolderLbl.grid(row=6, column=0, sticky='E', padx=5, pady=2)
        self.TensorBoard_inFolderTxt = Entry(self.tb_frame )
        self.TensorBoard_inFolderTxt.grid(row=6,
                              column=1,
                              columnspan=5,
                              sticky="W",
                              pady=3)
        self.TensorBoard_inFileBtn = Button(self.tb_frame, text="Browse ...",
                                command=self.load_tensorBoard_path)
        self.TensorBoard_inFileBtn.grid(row=6,
                            column=9,
                            columnspan=1,
                            sticky='W',
                            padx=5,
                            pady=1)
        self.portTagLbl = Label(self.tb_frame, text="Port:")
        self.portTagLbl.grid(row=7, column=0, sticky='E', padx=5, pady=2)
        self.portTxt = Entry(self.tb_frame,
                              textvariable=self.setting_PORT_mod)
        self.portTxt.grid(row=7, column=1, columnspan=1, sticky="W", pady=1)


        self.TensorBoardBtn = Button(self.tb_frame,
                                  state='disabled',
                                  text="Start TensorBoard",
                                  command=self.start_tensorBoard)
        self.TensorBoardBtn.grid(row=8, column=0, sticky='W', padx=1, pady=1)


        # setting input modalities: FLAIR + T1 are mandatory
        # Mod 3 / 4 are optional
        self.flairTagLbl = Label(self.tr_frame, text="FLAIR modality:")
        self.flairTagLbl.grid(row=1, column=0, sticky='E', padx=5, pady=2)
        self.flairTxt = Entry(self.tr_frame,
                              textvariable=self.setting_FLAIR_mod)
        self.flairTxt.grid(row=1, column=1, columnspan=1, sticky="W", pady=1)

        self.t1TagLbl = Label(self.tr_frame, text="T1 modality:")
        self.t1TagLbl.grid(row=2, column=0, sticky='E', padx=5, pady=2)
        self.t1Txt = Entry(self.tr_frame, textvariable=self.setting_T1_mod)
        self.t1Txt.grid(row=2, column=1, columnspan=1, sticky="W", pady=1)

        self.mod3TagLbl = Label(self.tr_frame, text="Imaging modality 3:")
        self.mod3TagLbl.grid(row=3, column=0, sticky='E', padx=5, pady=2)
        self.mod3Txt = Entry(self.tr_frame,
                              textvariable=self.setting_MOD3_mod)
        self.mod3Txt.grid(row=3, column=1, columnspan=1, sticky="W", pady=1)

        self.mod4TagLbl = Label(self.tr_frame, text="Imaging modality 4:")
        self.mod4TagLbl.grid(row=4, column=0, sticky='E', padx=5, pady=2)
        self.mod4Txt = Entry(self.tr_frame,
                              textvariable=self.setting_MOD4_mod)
        self.mod4Txt.grid(row=4, column=1, columnspan=1, sticky="W", pady=1)

        self.maskTagLbl = Label(self.tr_frame, text="Imaging modality 5:")
        self.maskTagLbl.grid(row=5, column=0,
                             sticky='E', padx=5, pady=2)
        self.maskTxt = Entry(self.tr_frame, textvariable=self.setting_mask_mod)
        self.maskTxt.grid(row=5, column=1, columnspan=1, sticky="W", pady=1)

        ######
        self.lable1TagLbl = Label(self.tr_frame, text="Manual label 1:")
        self.lable1TagLbl.grid(row=1, column=2, sticky='E', padx=5, pady=2)
        self.lable1TagLbltxt = Entry(self.tr_frame,
                              textvariable=self.setting_label1_mod)
        self.lable1TagLbltxt.grid(row=1, column=3, columnspan=1, sticky="W", pady=1)

        self.lable2Lbl = Label(self.tr_frame, text="Manual label 2:")
        self.lable2Lbl.grid(row=2, column=2, sticky='E', padx=5, pady=2)
        self.lable2Lbltxt = Entry(self.tr_frame, textvariable=self.setting_label2_mod)
        self.lable2Lbltxt.grid(row=2, column=3, columnspan=1, sticky="W", pady=1)

        self.lablel3bl = Label(self.tr_frame, text="Manual label 3:")
        self.lablel3bl.grid(row=3, column=2, sticky='E', padx=5, pady=2)
        self.lablel3bltxt = Entry(self.tr_frame,
                              textvariable=self.setting_label3_mod)
        self.lablel3bltxt.grid(row=3, column=3, columnspan=1, sticky="W", pady=1)

        self.lablel4Lbl = Label(self.tr_frame, text="Manual label 4:")
        self.lablel4Lbl.grid(row=4, column=2, sticky='E', padx=5, pady=2)
        self.lablel4LblTxt = Entry(self.tr_frame,
                              textvariable=self.setting_label4_mod)
        self.lablel4LblTxt.grid(row=4, column=3, columnspan=1, sticky="W", pady=1)

        self.lablel5Lbl = Label(self.tr_frame, text="Manual label 5:")
        self.lablel5Lbl.grid(row=5, column=2,
                             sticky='E', padx=5, pady=2)
        self.lablel5LblTxt = Entry(self.tr_frame, textvariable=self.setting_label5_mod)
        self.lablel5LblTxt.grid(row=5, column=3, columnspan=1, sticky="W", pady=1)
        ######

        self.Number_of_classes = Label(self.tr_frame, text="Number of classes:")
        self.Number_of_classes.grid(row=6, column=2, sticky="W")
        self.Number_of_classes_ent = Entry(self.tr_frame,
                               textvariable=self.setting_Number_of_classes)
        self.Number_of_classes_ent.grid(row=6, column=3, columnspan=1, sticky="W", pady=1)

        # model settings
        self.Annotation = Checkbutton(self.model_frame,
                                    text="Annotation Network",
                                    var=self.setting_Annotation)
        self.Annotation.grid(row=7, column=0, sticky='W', padx=5,
                                  pady=2)

        self.Segmentation = Checkbutton(self.model_frame,
                                    text="Segmentation Network",
                                    var=self.setting_Segmentation)
        self.Segmentation.grid(row=7, column=1, sticky='W', padx=5,
                                  pady=2)



        self.modelTagLbl = Label(self.model_frame, text="CNN model name:")
        self.modelTagLbl.grid(row=8, column=0,
                              sticky='E', padx=5, pady=2)
        self.modelTxt = Entry(self.model_frame,
                              textvariable=self.setting_net_name)
        self.modelTxt.grid(row=8, column=1, columnspan=1, sticky="W", pady=1)

        self.checkPretrain = Checkbutton(self. model_frame,
                                         text="learned model",
                                         var=self.setting_use_pretrained_model)
        self.checkPretrain.grid(row=8, column=3, padx=5, pady=5)

        self.learned_models()

        self.pretrainTxt = OptionMenu(self.model_frame,
                                      self.setting_pretrained_model,
                                      *self.list_train_pretrained_nets)
        self.pretrainTxt.grid(row=8, column=5, sticky='E', padx=5, pady=5)



        # START button links
        self.trainingBtn = Button(self.train_frame,
                                  state='disabled',
                                  text="Start training",
                                  command=self.training)
        self.trainingBtn.grid(row=9, column=0, sticky='W', padx=1, pady=1)


        self.bin_seg = Checkbutton(self.train_frame,
                                    text="Binary Segmentation Network",
                                    var=self.setting_bin)
        self.bin_seg.grid(row=7, column=0, sticky='W', padx=(120, 1),
                                  pady=1)

        self.multi_seg = Checkbutton(self.train_frame,
                                    text="Multi Class Segmentation Network",
                                    var=self.setting_multi)
        self.multi_seg.grid(row=7, column=0, sticky='W', padx=(325, 1),
                                  pady=1)

        self.withmnistdataset = Checkbutton(self.train_frame,
                                    text="Training with MNIST dataset",
                                    var=self.setting_Mnist_dataset)
        self.withmnistdataset.grid(row=9, column=0, sticky='W', padx=(120, 1),
                                  pady=1)

        self.withmsdataset = Checkbutton(self.train_frame,
                                    text="Training with MS dataset",
                                    var=self.setting_MS_dataset)
        self.withmsdataset.grid(row=9, column=0, sticky='W', padx=(325, 1),
                                  pady=1)

        img1 = ImageTk.PhotoImage(Image.open('images/Drawing1.jpg'))
        imglabel = Label(self.train_frame, image=img1)
        imglabel.image = img1
        imglabel.grid(row=10, column=0, padx=1, pady=1)



        # --------------------------------------------------
        # inference tab
        # --------------------------------------------------
        self.tt_frame = LabelFrame(self.test_frame, text="Inference images:")
        self.tt_frame.grid(row=0, columnspan=cl_s, sticky='WE',
                           padx=5, pady=5, ipadx=5, ipady=5)
        self.test_model_frame = LabelFrame(self.test_frame, text="CNN_GUI model:")
        self.test_model_frame.grid(row=5, columnspan=cl_s, sticky='WE',
                                   padx=5, pady=5, ipadx=5, ipady=5)

        # testing settings
        self.test_inFolderLbl = Label(self.tt_frame, text="Testing folder:")
        self.test_inFolderLbl.grid(row=0, column=0, sticky='E', padx=5, pady=2)
        self.test_inFolderTxt = Entry(self.tt_frame)
        self.test_inFolderTxt.grid(row=0,
                                   column=1,
                                   columnspan=5,
                                   sticky="W",
                                   pady=3)
        self.test_inFileBtn = Button(self.tt_frame, text="Browse ...",
                                     command=self.testing_path)
        self.test_inFileBtn.grid(row=0,
                                 column=5,
                                 columnspan=1,
                                 sticky='W',
                                 padx=5,
                                 pady=1)

        self.test_settingsBtn = Button(self.tt_frame,
                                      text="Settings",
                                      command=self.Add_Settings)
        self.test_settingsBtn.grid(row=0,
                                  column=10,
                                  columnspan=1,
                                  sticky="W",
                                  padx=(100, 1),
                                  pady=1)

        self.test_flairTagLbl = Label(self.tt_frame, text="FLAIR modality:")
        self.test_flairTagLbl.grid(row=1, column=0, sticky='E', padx=5, pady=2)
        self.test_flairTxt = Entry(self.tt_frame,
                              textvariable=self.setting_FLAIR_mod)
        self.test_flairTxt.grid(row=1, column=1, columnspan=1, sticky="W", pady=1)

        self.test_t1TagLbl = Label(self.tt_frame, text="T1 modality:")
        self.test_t1TagLbl.grid(row=2, column=0, sticky='E', padx=5, pady=2)
        self.test_t1Txt = Entry(self.tt_frame, textvariable=self.setting_T1_mod)
        self.test_t1Txt.grid(row=2, column=1, columnspan=1, sticky="W", pady=1)

        self.test_mod3TagLbl = Label(self.tt_frame, text="Imaging modality 3:")
        self.test_mod3TagLbl.grid(row=3, column=0, sticky='E', padx=5, pady=2)
        self.test_mod3Txt = Entry(self.tt_frame,
                              textvariable=self.setting_MOD3_mod)
        self.test_mod3Txt.grid(row=3, column=1, columnspan=1, sticky="W", pady=1)

        self.test_mod4TagLbl = Label(self.tt_frame, text="Imaging modality 4:")
        self.test_mod4TagLbl.grid(row=4, column=0, sticky='E', padx=5, pady=2)
        self.test_mod4Txt = Entry(self.tt_frame,
                              textvariable=self.setting_MOD4_mod)
        self.test_mod4Txt.grid(row=4, column=1, columnspan=1, sticky="W", pady=1)

        self.test_pretrainTxt = OptionMenu(self.test_model_frame,
                                           self.setting_inference_model,
                                           *self.list_test_nets)

        self.setting_inference_model.set('None')
        self.test_pretrainTxt.grid(row=5, column=0, sticky='E', padx=5, pady=5)
        self.checkBatchprediction = Checkbutton(self.test_model_frame,
                                    text="Batch Prediction",
                                    var=self.setting_batch_prediction)
        self.checkBatchprediction.grid(row=5, column=2, sticky='W') 

       

        # START button links cto docker task
        self.inferenceBtn = Button(self.test_frame,
                                   state='disabled',
                                   text="Start inference",
                                   command=self.inference)
        self.inferenceBtn.grid(row=7, column=0, sticky='W', padx=1, pady=1)

        img2 = ImageTk.PhotoImage(Image.open('images/Drawing1.jpg'))
        imglabel2 = Label(self.test_frame, image=img2)
        imglabel2.image = img2
        imglabel2.grid(row=8, column=0, padx=1, pady=5)

        # train / test ABOUT button
        self.train_aboutBtn = Button(self.train_frame,
                                     text="about",
                                     command=self.About)
        self.train_aboutBtn.grid(row=7,
                                 column=4,
                                 sticky='E',
                                 padx=(1, 1),
                                 pady=1)

        self.test_aboutBtn = Button(self.test_frame,
                                    text="about",
                                    command=self.About)
        self.test_aboutBtn.grid(row=7,
                                column=4,
                                sticky='E',
                                padx=(1, 1),
                                pady=1)

        # Processing state
        self.process_indicator = StringVar()
        self.process_indicator.set(' ')
        self.label_indicator = Label(master,
                                     textvariable=self.process_indicator)
        self.label_indicator.pack(side="left")

        # Closing processing events is implemented via
        # a master protocol
        self.master.protocol("WM_DELETE_WINDOW", self.close_event)

    def Add_Settings(self):
        """
        Setting other parameters using an emerging window
        CNN_GUI parameters, CUDA device, post-processing....

        """
        t = Toplevel(self.master)
        t.wm_title("Additional Parameter Settings")

        # data parameters
        t_data = LabelFrame(t, text="Pre/Post-Processing Settings")
        t_data.grid(row=0, sticky="WE")
        checkBias = Checkbutton(t_data,
                                    text="Bias correction",
                                    var=self.setting_bias_correction)
        checkBias.grid(row=0, sticky='W')

        self.biasTxt = OptionMenu(t_data,  self.setting_bias_choice,
                                      *self.list_bias)
        self.biasTxt.grid(row=0, column=1, sticky='E', padx=5, pady=5)




        Bias_par_iter_label = Label(t_data, text="Bias correction iteration number:")
        Bias_par_iter_label.grid(row=1, sticky="W")
        Bias_par_niter = Entry(t_data,
                                textvariable=self.setting_Bias_cor_niter)
        Bias_par_niter.grid(row=1, column=1, sticky="E")



        Bias_par_smooth_label = Label(t_data, text="Bias correction smooth:")
        Bias_par_smooth_label.grid(row=2, sticky="W")
        Bias_par_smooth = Entry(t_data,
                                    textvariable=self.setting_Bias_cor_smooth)
        Bias_par_smooth.grid(row=2, column=1, sticky="E")

        Bias_par_type_label = Label(t_data, text="Bias correction type: 1 = T1w, 2 = T2w, 3 = PD ")
        Bias_par_type_label.grid(row=3, sticky="W")
        Bias_par_type = Entry(t_data,
                                    textvariable=self.setting_Bias_cor_type)
        Bias_par_type.grid(row=3, column=1, sticky="E")





        checkPretrain = Checkbutton(t_data,
                                    text="Registration",
                                    var=self.setting_register_modalities)


        checkPretrain.grid(row=4, sticky='W')

        # register_label = Label(t_data, text="register mod:(FlairtoT1 or T1toFlair)")
        register_label = Label(t_data, text="Register Mod. to T1, Flair or Std. Space:")
        register_label.grid(row=5, sticky="W")
        # register_label_entry = Entry(t_data, textvariable=self.setting_register_modalities_Kind)
        # register_label_entry.grid(row=1, column=1, sticky="E")
        self.regTxt = OptionMenu(t_data,  self.setting_reg_space,
                                      *self.list_standard_space_reg_list)
        self.regTxt.grid(row=5, column=1, sticky='E', padx=5, pady=5)


        checkSkull = Checkbutton(t_data,
                                 text="Skull Striping",
                                 var=self.setting_skull_strippingping)
        checkSkull.grid(row=6, sticky="W")
        checkDenoise = Checkbutton(t_data,
                                   text="Denoising",
                                   var=self.setting_denoise)
        checkDenoise.grid(row=7, sticky="W")

        denoise_iter_label = Label(t_data, text=" Denoise iter:               ")
        denoise_iter_label.grid(row=8, sticky="W")
        denoise_iter_entry = Entry(t_data, textvariable=self.setting_denoise_iter)
        denoise_iter_entry.grid(row=8, column=1, sticky="E")

        check_tmp = Checkbutton(t_data,
                                text="Save tmp files",
                                var=self.setting_save_tmp)
        check_tmp.grid(row=9, sticky="W")
        checkdebug = Checkbutton(t_data,
                                 text="Debug mode",
                                 var=self.setting_debug)
        checkdebug.grid(row=10, sticky="W")


        t_bin_label = Label(t_data, text="Threshold:      ")
        t_bin_label.grid(row=11, sticky="W")
        t_bin_entry = Entry(t_data, textvariable=self.setting_t_bin)
        t_bin_entry.grid(row=11, column=1, sticky="E")

        l_min_label = Label(t_data, text="Output Volume Tolerance:         ")
        l_min_label.grid(row=12, sticky="W")
        l_min_entry = Entry(t_data, textvariable=self.setting_l_min)
        l_min_entry.grid(row=12, column=1, sticky="E")

        vol_min_label = Label(t_data, text="Error Tolerance:   ")
        vol_min_label.grid(row=13, sticky="W")
        vol_min_entry = Entry(t_data, textvariable=self.setting_min_error)
        vol_min_entry.grid(row=13, column=1, sticky="E")



        # model parameters
        t_model = LabelFrame(t, text="Training:")
        t_model.grid(row=14, sticky="EW")

        maxepochs_label = Label(t_model, text="Max epochs:                  ")
        maxepochs_label.grid(row=15, sticky="W")
        maxepochs_entry = Entry(t_model, textvariable=self.setting_max_epochs)
        maxepochs_entry.grid(row=15, column=1, sticky="E")

        trainsplit_label = Label(t_model, text="Validation %:           ")
        trainsplit_label.grid(row=16, sticky="W")
        trainsplit_entry = Entry(t_model, textvariable=self.setting_train_split)
        trainsplit_entry.grid(row=16, column=1, sticky="E")

        batchsize_label = Label(t_model, text="Test batch size:")
        batchsize_label.grid(row=17, sticky="W")
        batchsize_entry = Entry(t_model, textvariable=self.setting_batch_size)
        batchsize_entry.grid(row=17, column=1, sticky="E")


        mode_label = Label(t_model, text="Verbosity:")
        mode_label.grid(row=18, sticky="W")
        mode_entry = Entry(t_model, textvariable=self.setting_net_verbose)
        mode_entry.grid(row=18, column=1, sticky="E")

        # gpu_mode = Checkbutton(t_model,
        #                         text="GPU:",
        # #                         var=self.setting_mode)
        # #gpu_mode.grid(row=10, sticky="W")

        gpu_number = Label(t_model, text="GPU number:")
        gpu_number.grid(row=19, sticky="W")
        gpu_entry = Entry(t_model, textvariable=self.setting_gpu_number)
        gpu_entry.grid(row=19, column=1, sticky="W")


        # # training parameters
        # tr_model = LabelFrame(t, text="Training:")
        # tr_model.grid(row=18, sticky="EW")

        balanced_label = Label(t_model, text="Balanced dataset:    ")
        balanced_label.grid(row=20, sticky="W")
        balanced_entry = Entry(t_model, textvariable=self.setting_balanced_dataset)
        balanced_entry.grid(row=20, column=1, sticky="E")

        fraction_label = Label(t_model, text="Fraction negative/positives Training: ")
        fraction_label.grid(row=21, sticky="W")
        fraction_entry = Entry(t_model, textvariable=self.setting_fract_negatives)
        fraction_entry.grid(row=21, column=1, sticky="E")

        fraction_label_cv = Label(t_model, text="Fraction negative/positives Cross Validation: ")
        fraction_label_cv.grid(row=22, sticky="W")
        fraction_entry_cv = Entry(t_model, textvariable=self.setting_fract_negatives_cv)
        fraction_entry_cv.grid(row=22, column=1, sticky="E")


    def load_tensorBoard_path(self):
        """
        Select training path from disk and write it.
        If the app is run inside a container,
        link the iniitaldir with /data
        """
        initialdir = '/data' if self.container else os.getcwd()
        fname = askdirectory(initialdir=initialdir)
        if fname:
            try:
                self.setting_tensorboard_folder.set(fname)
                self.TensorBoard_inFolderTxt.delete(0, END)
                self.TensorBoard_inFolderTxt.insert(0, self.setting_tensorboard_folder.get())
                self.TensorBoardBtn['state'] = 'normal'
            except:
                pass

    def read_default_configuration(self):


        # default_config = configparser.SafeConfigParser()
        default_config = configparser.ConfigParser()
        default_config.read(os.path.join(self.path, 'config', 'configuration.cfg'))

        # dastaset parameters
        self.setting_training_folder.set(default_config.get('TrainTestSet',
                                                          'train_folder'))
        self.setting_tensorboard_folder.set(default_config.get('tensorboard',
                                                          'tensorBoard_folder'))
        self.setting_PORT_mod.set(default_config.getint('tensorboard',
                                                     'port'))

        self.setting_test_folder.set(default_config.get('TrainTestSet',
                                                      'inference_folder'))
        self.setting_FLAIR_mod.set(default_config.get('TrainTestSet','flair_mods'))
        self.setting_T1_mod.set(default_config.get('TrainTestSet','t1_mods'))
        self.setting_MOD3_mod.set(default_config.get('TrainTestSet','mod3_mods'))
        self.setting_MOD4_mod.set(default_config.get('TrainTestSet','mod4_mods'))
        self.setting_mask_mod.set(default_config.get('TrainTestSet','roi_mods'))
        self.all_label.set(default_config.get('TrainTestSet','all_isolated_label'))
        self.setting_label1_mod.set(default_config.get('TrainTestSet','l1_mods'))
        self.setting_label2_mod.set(default_config.get('TrainTestSet','l2_mods'))
        self.setting_label3_mod.set(default_config.get('TrainTestSet','l3_mods'))
        self.setting_label4_mod.set(default_config.get('TrainTestSet','l4_mods'))
        self.setting_label5_mod.set(default_config.get('TrainTestSet','l5_mods'))

        self.model_1_train.set(default_config.get('completed', 'model_1_train'))
        self.model_2_train.set(default_config.get('completed', 'model_2_train'))
        self.pre_processing.set(default_config.get('completed', 'pre_processing'))


        self.setting_register_modalities.set(default_config.get('TrainTestSet', 'register_modalities'))
        self.setting_bias_correction.set(default_config.get('TrainTestSet', 'bias_correction'))
        self.setting_Number_of_classes.set(default_config.get('TrainTestSet', 'number_of_classes'))
        self.setting_batch_prediction.set(default_config.get('TrainTestSet', 'batch_prediction'))

        self.setting_Mnist_dataset.set(default_config.get('TrainTestSet', 'mnist_dataset'))
        self.setting_MS_dataset.set(default_config.get('TrainTestSet', 'ms_dataset'))


        self.setting_Annotation.set(default_config.get('TrainTestSet', 'Annotation_network'))
        self.setting_Segmentation.set(default_config.get('TrainTestSet', 'Segmentation_network'))

        self.setting_bin.set(default_config.get('TrainTestSet', 'binary_network'))
        self.setting_multi.set(default_config.get('TrainTestSet', 'multi_class_network'))


        # self.setting_batch_prediction
        # self.setting_register_modalities_Kind.set(default_config.get('TrainTestSet', 'register_modalities_Kind'))self
        self.setting_denoise.set(default_config.get('TrainTestSet', 'denoise'))
        self.setting_denoise_iter.set(default_config.getint('TrainTestSet', 'denoise_iter'))
        # self.setting_Bias_cor_niter = IntVar()
        # self.setting_Bias_cor_smooth = IntVar()
        # self.setting_Bias_cor_type = IntVar()
        self.setting_Bias_cor_niter.set(default_config.getint('TrainTestSet', 'bias_iter'))
        self.setting_bias_choice.set(default_config.get('TrainTestSet', 'bias_choice'))

        self.setting_Bias_cor_smooth.set(default_config.getint('TrainTestSet', 'bias_smooth'))
        self.setting_Bias_cor_type.set(default_config.getint('TrainTestSet', 'bias_type'))


        self.setting_skull_strippingping.set(default_config.get('TrainTestSet', 'skull_strippingping'))
        self.setting_save_tmp.set(default_config.get('TrainTestSet', 'save_tmp'))
        self.setting_debug.set(default_config.get('TrainTestSet', 'debug'))

        # train parameters
        self.setting_use_pretrained_model.set(default_config.get('TrainTestSet', 'full_train'))
        self.setting_pretrained_model.set(default_config.get('TrainTestSet', 'pretrained_model'))
        self.setting_reg_space.set(default_config.get('TrainTestSet', 'reg_space'))
        # ///////
        self.setting_pretrained_model.set(default_config.get('TrainTestSet', 'pretrained_model'))

        self.setting_inference_model.set("      ")
        self.setting_balanced_dataset.set(default_config.get('TrainTestSet', 'balanced_training'))
        self.setting_fract_negatives.set(default_config.getfloat('TrainTestSet', 'fraction_negatives'))
        self.setting_fract_negatives_cv.set(default_config.getfloat('TrainTestSet', 'fraction_negatives_CV'))

        # model parameters
        self.setting_net_folder = os.path.join(self.current_folder, 'nets')
        self.setting_net_name.set(default_config.get('TrainTestSet', 'name'))
        self.setting_train_split.set(default_config.getfloat('TrainTestSet', 'train_split'))
        self.setting_max_epochs.set(default_config.getint('TrainTestSet', 'max_epochs'))
        self.setting_patience.set(default_config.getint('TrainTestSet', 'patience'))
        self.setting_batch_size.set(default_config.getint('TrainTestSet', 'batch_size'))
        self.setting_net_verbose.set(default_config.get('TrainTestSet', 'net_verbose'))
        self.setting_gpu_number.set(default_config.getint('TrainTestSet', 'gpu_number'))
        # self.setting_mode.set(default_config.get('model', 'gpu_mode'))

        # post-processing
        self.setting_l_min.set(default_config.getint('TrainTestSet',
                                                   'l_min'))
        self.setting_t_bin.set(default_config.getfloat('TrainTestSet',
                                                     't_bin'))
        self.setting_min_error.set(default_config.getfloat('TrainTestSet',
                                                     'min_error'))

    def write_default_configuration(self):

        user_new_settings = configparser.ConfigParser()

        # dataset parameters
        user_new_settings.add_section('TrainTestSet')
        user_new_settings.set('TrainTestSet', 'Annotation_network', str(self.setting_Annotation.get()))
        user_new_settings.set('TrainTestSet', 'Segmentation_network', str(self.setting_Segmentation.get()))
        user_new_settings.set('TrainTestSet', 'binary_network', str(self.setting_bin.get()))
        user_new_settings.set('TrainTestSet', 'multi_class_network', str(self.setting_multi.get()))


        user_new_settings.set('TrainTestSet', 'name', self.setting_net_name.get())
        user_new_settings.set('TrainTestSet', 'train_folder', self.setting_training_folder.get())
        user_new_settings.set('TrainTestSet', 'inference_folder', self.setting_test_folder.get())

        user_new_settings.set('TrainTestSet', 'all_isolated_label', self.all_label.get())

        user_new_settings.set('TrainTestSet', 'flair_mods', self.setting_FLAIR_mod.get())
        user_new_settings.set('TrainTestSet', 't1_mods', self.setting_T1_mod.get())
        user_new_settings.set('TrainTestSet', 'mod3_mods', self.setting_MOD3_mod.get())
        user_new_settings.set('TrainTestSet', 'mod4_mods', self.setting_MOD4_mod.get())
        user_new_settings.set('TrainTestSet', 'roi_mods', self.setting_mask_mod.get())

        user_new_settings.set('TrainTestSet', 'l1_mods', self.setting_label1_mod.get())
        user_new_settings.set('TrainTestSet', 'l2_mods', self.setting_label2_mod.get())
        user_new_settings.set('TrainTestSet', 'l3_mods', self.setting_label3_mod.get())
        user_new_settings.set('TrainTestSet', 'l4_mods', self.setting_label4_mod.get())
        user_new_settings.set('TrainTestSet', 'l5_mods', self.setting_label5_mod.get())

        user_new_settings.set('TrainTestSet', 'register_modalities', str(self.setting_register_modalities.get()))
        user_new_settings.set('TrainTestSet', 'bias_correction', str(self.setting_bias_correction.get()))
        user_new_settings.set('TrainTestSet', 'batch_prediction', str(self.setting_batch_prediction.get()))



        user_new_settings.set('TrainTestSet', 'mnist_dataset', str(self.setting_Mnist_dataset.get()))
        user_new_settings.set('TrainTestSet', 'ms_dataset', str(self.setting_MS_dataset.get()))


        # self.setting_batch_prediction.set(default_config.get('TrainTestSet', 'batch_prediction'))
        # user_new_settings.set('TrainTestSet', 'register_modalities_Kind', str(self.setting_register_modalities_Kind.get()))
        user_new_settings.set('TrainTestSet', 'reg_space', str(self.setting_reg_space.get()))
        user_new_settings.set('TrainTestSet', 'denoise', str(self.setting_denoise.get()))
        user_new_settings.set('TrainTestSet', 'denoise_iter', str(self.setting_denoise_iter.get()))

        # self.setting_Bias_cor_niter.set(default_config.getint('TrainTestSet', 'bias_iter'))
        # self.setting_Bias_cor_smooth.set(default_config.getint('TrainTestSet', 'bias_smooth'))
        # self.setting_Bias_cor_type.set(default_config.getint('TrainTestSet', 'bias_type'))
        user_new_settings.set('TrainTestSet', 'bias_iter', str(self.setting_Bias_cor_niter.get()))
        user_new_settings.set('TrainTestSet', 'number_of_classes', str(self.setting_Number_of_classes.get()))
        user_new_settings.set('TrainTestSet', 'bias_smooth', str(self.setting_Bias_cor_smooth.get()))   
        user_new_settings.set('TrainTestSet', 'bias_type', str(self.setting_Bias_cor_type.get())) 
        user_new_settings.set('TrainTestSet', 'bias_choice', str(self.setting_bias_choice.get()))    
        


        user_new_settings.set('TrainTestSet', 'skull_strippingping', str(self.setting_skull_strippingping.get()))
        user_new_settings.set('TrainTestSet', 'save_tmp', str(self.setting_save_tmp.get()))
        user_new_settings.set('TrainTestSet', 'debug', str(self.setting_debug.get()))


        user_new_settings.set('TrainTestSet',
                        'full_train',
                        str(not(self.setting_use_pretrained_model.get())))
        user_new_settings.set('TrainTestSet',
                        'pretrained_model',
                        str(self.setting_pretrained_model.get()))
        user_new_settings.set('TrainTestSet',
                        'balanced_training',
                        str(self.setting_balanced_dataset.get()))
        user_new_settings.set('TrainTestSet',
                        'fraction_negatives',
                        str(self.setting_fract_negatives.get()))

        user_new_settings.set('TrainTestSet',
                        'fraction_negatives_CV',
                        str(self.setting_fract_negatives_cv.get()))



        user_new_settings.set('TrainTestSet', 'pretrained', str(self.setting_pretrained))
        user_new_settings.set('TrainTestSet', 'train_split', str(self.setting_train_split.get()))
        user_new_settings.set('TrainTestSet', 'max_epochs', str(self.setting_max_epochs.get()))
        user_new_settings.set('TrainTestSet', 'patience', str(self.setting_patience.get()))
        user_new_settings.set('TrainTestSet', 'batch_size', str(self.setting_batch_size.get()))
        user_new_settings.set('TrainTestSet', 'net_verbose', str(self.setting_net_verbose.get()))
        # user_new_settings.set('model', 'gpu_mode', self.setting_mode.get())
        user_new_settings.set('TrainTestSet', 'gpu_number', str(self.setting_gpu_number.get()))
        user_new_settings.set('TrainTestSet', 't_bin', str(self.setting_t_bin.get()))
        user_new_settings.set('TrainTestSet', 'l_min', str(self.setting_l_min.get()))
        user_new_settings.set('TrainTestSet',
                        'min_error', str(self.setting_min_error.get()))
        user_new_settings.add_section('tensorboard')
        user_new_settings.set('tensorboard', 'port', str(self.setting_PORT_mod.get()))
        user_new_settings.set('tensorboard', 'tensorBoard_folder', self.setting_tensorboard_folder.get())

        user_new_settings.add_section('completed')
        user_new_settings.set('completed', 'model_1_train', str(self.model_1_train.get()))
        user_new_settings.set('completed', 'model_2_train', str(self.model_2_train.get()))
        user_new_settings.set('completed', 'pre_processing', str(self.pre_processing.get()))

        # Writing our configuration file to 'example.cfg'
        with open(os.path.join(self.path,
                               'config',
                               'configuration.cfg'), 'w') as configfile:


            user_new_settings.write(configfile)


             # user_new_settings.readfp(configfile)
             # configfile.write(user_new_settings)
            # user_new_settings.write(bytes(os.path.join(self.path,
            #                    'config',
            #                    'configuration.cfg'), 'UTF-8'y))

            # plaintext = input("Please enter the text you want to compress")
            # filename = input("Please enter the desired filename")
            # with gzip.open(filename + ".gz", "wb") as outfile:
            #     outfile.write(bytes(plaintext, 'UTF-8'))

    def training_path(self):

        initialdir = '/data' if self.container else os.getcwd()
        fname = askdirectory(initialdir=initialdir)
        if fname:
            try:
                self.setting_training_folder.set(fname)
                self.inFolderTxt.delete(0, END)
                self.inFolderTxt.insert(0, self.setting_training_folder.get())
                self.trainingBtn['state'] = 'normal'
            except:
                pass

    def testing_path(self):

        initialdir = '/data' if self.container else os.getcwd()
        fname = askdirectory(initialdir=initialdir)
        if fname:
            try:
                self.setting_test_folder.set(fname)
                self.test_inFolderTxt.delete(0, END)
                self.test_inFolderTxt.insert(0, self.setting_test_folder.get())
                self.inferenceBtn['state'] = 'normal'
            except:
                pass

    def learned_models(self):
        folders = os.listdir(self.setting_net_folder)
        self.list_train_pretrained_nets = folders
        self.list_test_nets = folders

    def write_to_console(self, txt):
        self.command_out.insert(END, str(txt))

    def write_to_test_console(self, txt):
        self.test_command_out.insert(END, str(txt))

    def start_tensorBoard(self):
            """
            Method implementing the training process:
            - write the configuration to disk
            - Run the process on a new thread
            """
            try:
                if self.setting_PORT_mod.get() == None:
                    print("\n")
            except ValueError:
                print("ERROR: Port number and TensorBoard folder must be defined  before starting...\n")
                return

            self.TensorBoardBtn['state'] = 'disable'

            if self.setting_PORT_mod.get() is not None:
                # self.TensorBoardBtn['state'] = 'normal'
                print("\n-----------------------")
                print("Starting TensorBoard ...")
                print("TensorBoard folder:", self.setting_tensorboard_folder.get(), "\n")
                thispath = self.setting_tensorboard_folder.get()
                thisport = self.setting_PORT_mod.get()
                self.write_default_configuration()
                print("The port for TensorBoard is set to be:", thisport)
                # import appscript
                pp = os.path.join(self.path, 'spider', 'bin')

                CURRENT_PATHx = os.path.split(os.path.realpath(__file__))[0]
                # tensorboard = CURRENT_PATHx + '/libs/bin/tensorboard'
                Folder=thispath
                Port=thisport
                os_host = platform.system()
                if os_host == 'Windows':
                    arg1 = ' ' + '--logdir  ' + str(Folder) + ' ' + '  --port  ' + str(Port)
                    os.system("start cmd  /c   'tensorboard   {}'".format(arg1))
                elif os_host == 'Linux':
                    arg1 =str(Folder)+'  ' + str(Port)
                    os.system("dbus-launch gnome-terminal -e 'bash -c \"bash  tensorb.sh   {}; exec bash\"'".format(arg1))

                elif os_host == 'Darwin':
                    import appscript
                    appscript.app('Terminal').do_script(
                        'tensorboard     --logdir=' + str(
                            thispath) + '  --port=' + str(thisport))

                else:
                    print("> ERROR: The OS system", os_host, "is not currently supported.")


    def inference(self):


        if self.setting_inference_model.get() == 'None':
            print("ERROR: Please, select a network model before starting...\n")
            return
        if self.test_task is None:
            self.inferenceBtn.config(state='disabled')
            self.setting_net_name.set(self.setting_inference_model.get())
            self.setting_use_pretrained_model.set(False)
            self.write_default_configuration()
            print("\n-----------------------")
            print("Running configuration:")
            print("-----------------------")
            print("Inference model:", self.setting_model_mod.get())
            print("Inference folder:", self.setting_test_folder.get(), "\n")

            print("Method info:")
            print("------------")
            self.test_task = Threaded_run(self.write_to_test_console,
                                          self.test_queue, mode='testing')
            self.test_task.start()
            self.master.after(100, self.running_state)

    def training(self):
        """
        Method implementing the training process:
        - write the configuration to disk
        - Run the process on a new thread
        """

        if self.setting_net_name.get() == 'None' or self.setting_net_name.get() == '':

            print(CRED +"ERROR:"+ CEND)
            print('\x1b[6;30;41m' + "Please define network name before starting..." + '\x1b[0m')
            print("\n")

            return

        if self.setting_Mnist_dataset.get() is False and self.setting_MS_dataset.get() is False:
            print(CRED + "ERROR:" + CEND)
            print('\x1b[6;30;41m' + "Please set training dataset(mnist or ms datset) before starting..." + '\x1b[0m')
            print("\n")
            return

        if self.setting_Mnist_dataset.get() is  True and self.setting_MS_dataset.get() is True:
            print(CRED + "ERROR:" + CEND)
            print('\x1b[6;30;41m' + "Training   with mnist and ms datset simultaneously is not possible!,"
                                    " please set correctly training dataset before starting..." + '\x1b[0m')
            print("\n")

            return


        if self.setting_Segmentation.get() is False and self.setting_Annotation.get() is False:
            print(CRED + "ERROR:" + CEND)
            print('\x1b[6;30;41m' + "Please set network to Segmentation or Annotation before starting..." + '\x1b[0m')
            print("\n")
            return

        if self.setting_Segmentation.get() is  True and self.setting_Annotation.get() is True:
            print(CRED + "ERROR:" + CEND)
            print('\x1b[6;30;41m' + "Working with Segmentation and Annotation network simultaneously is not possible!,"
                                    " please set correctly network before starting..." + '\x1b[0m')
            print("\n")

            return



        if self.setting_bin.get() is False and self.setting_multi.get() is False:
            print(CRED + "ERROR:" + CEND)
            print('\x1b[6;30;41m' + "Please set network to binary or multi class segmentation before starting..." + '\x1b[0m')
            print("\n")
            return

        if self.setting_bin.get() is  True and self.setting_multi.get() is True:
            print(CRED + "ERROR:" + CEND)
            print('\x1b[6;30;41m' + "Training  binary and multi class segmentation network simultaneously is not possible!,"
                                    "please set correctly network before starting..." + '\x1b[0m')
            print("\n")

            return

        if self.setting_bin.get() is True and self.setting_Number_of_classes.get() != 1:
            print(CRED + "ERROR:" + CEND)
            print('\x1b[6;30;41m' + "Training  with class not equal to one for binary segmentation is not possible!, "
                                    "please set correctly number of classes before starting..." + '\x1b[0m')
            print("\n")

            return

        if self.setting_Number_of_classes.get() == 0 or self.setting_Number_of_classes.get() < 0:
            print(CRED + "ERROR:" + CEND)
            print('\x1b[6;30;41m' + "Traninig  with zero classes is not possible!, "
                                    "please set correctly number of classes before starting..." + '\x1b[0m')
            print("\n")

            return



        self.trainingBtn['state'] = 'disable'



        if self.train_task is None:
            self.trainingBtn.update()
            self.write_default_configuration()
            print("\n-----------------------")
            print("Running configuration:")
            print("-----------------------")
            print("Train model:", self.setting_net_name.get())
            print("Training folder:", self.setting_training_folder.get(), "\n")

            print("Method info:")
            print("------------")

            self.train_task = Threaded_run(self.write_to_console,
                                           self.test_queue,
                                           mode='training')
            self.train_task.start()
            self.master.after(100, self.running_state)

    def About(self):

        t = Toplevel(self.master, width=500, height=500)
        t.wm_title("About the software and version number")

        # NIC logo + name
        title = Label(t,
                      text="Multi-Label Image Segmentation version (" + self.version + ")\n"
                      "An End-to-end Lesion Segmentation via Learning from Noisy Labels")
        title.grid(row=2, column=1, padx=20, pady=10)
        img = ImageTk.PhotoImage(Image.open('images/n.jpg'))
        # img = img.resize((250, 250), Image.ANTIALIAS)
        imglabel = Label(t, image=img)
        imglabel.image = img
        imglabel.grid(row=1, column=1, padx=10, pady=10)
        group_name = Label(t,
                           text="Kevin Bronik and Le Zhang (2020) \n " +
                           "Medical Physics and Biomedical Engineering, UCL")
        group_name.grid(row=3, column=1)

    def running_state(self):

        self.process_indicator.set('Running... please wait - if you want to stop program, just close the GUI')
        try:
            msg = self.test_queue.get(0)
            self.process_indicator.set('Done!')
            self.inferenceBtn['state'] = 'normal'
            self.trainingBtn['state'] = 'normal'
        except queue.Empty:
            self.master.after(100, self.running_state)

    def close_event(self):
        """
        Stop the thread processes using OS related calls.
        """
        if self.train_task is not None:
            self.train_task.stop_process()
        if self.test_task is not None:
            self.test_task.stop_process()
        os.system('cls' if platform.system == "Windows" else 'clear')
        root.destroy()


class Threaded_run(threading.Thread):

    def __init__(self, print_func, queue, mode):
        threading.Thread.__init__(self)
        self.queue = queue
        self.mode = mode
        self.print_func = print_func
        self.process = None

    def run(self):

        settings = read_default_config()
        if self.mode == 'training':
            if settings['segmentation_network'] is True:
                print('\x1b[6;30;41m' + "                                          " + '\x1b[0m')
                print('\x1b[6;30;41m' + "Starting segmentation network training ..." + '\x1b[0m')
                print('\x1b[6;30;41m' + "                                          " + '\x1b[0m')
                trainingwork(settings)
            elif settings['annotation_network'] is True:
                print('\x1b[6;30;41m' + "                                        " + '\x1b[0m')
                print('\x1b[6;30;41m' + "Starting annotation network training ..." + '\x1b[0m')
                print('\x1b[6;30;41m' + "                                        " + '\x1b[0m')
                trainingwork_annotation(settings)
            else:
                print('training can not be done!')

        else:
            if settings['segmentation_network'] is True:
                print('\x1b[6;30;41m' + "                                           " + '\x1b[0m')
                print('\x1b[6;30;41m' + "Starting segmentation network inference ..." + '\x1b[0m')
                print('\x1b[6;30;41m' + "                                           " + '\x1b[0m')
                inference(settings)
            elif settings['annotation_network'] is True:
                print('\x1b[6;30;41m' + "                                         " + '\x1b[0m')
                print('\x1b[6;30;41m' + "Starting annotation network inference ..." + '\x1b[0m')
                print('\x1b[6;30;41m' + "                                         " + '\x1b[0m')
                inference_annotation(settings)
            else:
                print('inference can not be done!')

        self.queue.put(" ")

    def stop_process(self):
        try:
            if platform.system() == "Windows" :
                subprocess.Popen("taskkill /F /T /PID %i" % os.getpid() , shell=True)
            else:
                os.killpg(os.getpgid(self.process.pid), signal.SIGKILL)
        except:
            os.kill(os.getpid(), signal.SIGTERM)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--docker',
                        dest='docker',
                        action='store_true')
    parser.set_defaults(docker=False)
    args = parser.parse_args()
    root = Tk()
    root.resizable(width=False, height=False)
    run_CNN_GUI = CNN_GUI(root, args.docker)
    root.mainloop()
