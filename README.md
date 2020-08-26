[![GitHub issues](https://img.shields.io/github/issues/UCLBrain/MSLS)](https://github.com/UCLBrain/MSLS/issues)
[![GitHub forks](https://img.shields.io/github/forks/UCLBrain/MSLS)](https://github.com/UCLBrain/MSLS/network)
[![GitHub stars](https://img.shields.io/github/stars/UCLBrain/MSLS)](https://github.com/UCLBrain/MSLS/stargazers)
[![GitHub license](https://img.shields.io/github/license/UCLBrain/MSLS)](https://github.com/UCLBrain/MSLS/blob/master/LICENSE)


# Multi-Label Multi/Single-Class Image Segmentation


<br>
 <img height="510" src="images/diag.png"/>
</br>

# Publication
Le Zhang, Ryutaro Tanno, Kevin Bronik, Chen Jin, Parashkev Nachev, Frederik Barkhof, Olga Ciccarelli, and Daniel C. Alexander, Learning to Segment When Experts Disagree, International Conference on Medical image computing and Computer-Assisted Intervention (MICCAI). Springer, Cham, 2020.


<br>
 <img align="center" height="1000" src="images/Miccai_2020_abs.jpg"/>
</br>


<object data="https://github.com/UCLBrain/MSLS/MICCAI_2020.pdf" type="application/pdf" width="700px" height="700px">
    <embed src="https://github.com/UCLBrain/MSLS/MICCAI_2020.pdf">
        <p>Click here to see full pdf file: <a href="https://github.com/UCLBrain/MSLS/blob/master/MICCAI_2020.pdf">Link to PDF</a></p>
    </embed>
</object>



# Running the GUI Program! 

First, user needs to install Anaconda https://www.anaconda.com/

Then


```sh
  - conda env create -f conda_environment_Training_Inference.yml  
``` 
and 

```sh
  - conda activate traintestenv  
``` 
finally

```sh
  - python  Training_Inference_GUI.py 
``` 

After lunching the graphical user interface, user will need to provide necessary information to start training/testing as follows:  

<br>
 <img height="510" src="images/GUI.jpg" />
</br>


# Running the Program from the command line!

First 

```sh
  - conda activate traintestenv  
``` 
then for training


```sh
  - python  segmentation_network_Training_without_GUI.py  [or annotation_network_Training_without_GUI.py]
``` 

for testing

```sh
  - python  segmentation_network_Inference_without_GUI.py  [or annotation_network_Inference_without_GUI.py]
``` 

# Testing the Program!

<br>
 <img height="510" src="images/bin_seg_ex.jpg" />
</br>

..................................................................................................................................................................

<br>
 <img height="510" src="images/multi_seg_ex.jpg" />
</br>





