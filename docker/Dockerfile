FROM ubuntu:18.04
# FROM phusion/baseimage


MAINTAINER thisgithub

ENV USER mpirun

ENV DEBIAN_FRONTEND=noninteractive \
    HOME=/home/${USER} 
    
RUN apt-get update && apt-get install -y --no-install-recommends apt-utils
     #   libfreetype6-dev \
      #  libhdf5-serial-dev \
     #   libzmq3-dev \
     #   pkg-config \
     #   software-properties-common \
     #   unzip
RUN apt-get update -y && \
    apt-get install -y --no-install-recommends sudo apt-utils && \
    apt-get install -y --no-install-recommends openssh-server \
        python3-tk python-dev python-numpy python-pip python-virtualenv python-scipy \
        bzip2 g++ git curl unzip graphviz libgl1-mesa-glx libhdf5-dev wget \
        libfreetype6-dev libhdf5-serial-dev libzmq3-dev pkg-config software-properties-common unzip \ 
        gcc gfortran libopenmpi-dev openmpi-bin openmpi-common openmpi-doc binutils && \
    apt-get clean && apt-get purge && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*
# Some TF tools expect a "python" binary
 

RUN mkdir /var/run/sshd
RUN echo 'root:${USER}' | chpasswd
RUN sed -i 's/PermitRootLogin without-password/PermitRootLogin yes/' /etc/ssh/sshd_config

# SSH login fix. Otherwise user is kicked off after login
RUN sed 's@session\s*required\s*pam_loginuid.so@session optional pam_loginuid.so@g' -i /etc/pam.d/sshd

ENV NOTVISIBLE "in users profile"
RUN echo "export VISIBLE=now" >> /etc/profile

# ------------------------------------------------------------
# Add an 'mpirun' user
# ------------------------------------------------------------

RUN adduser --disabled-password --gecos "" ${USER} && \
    echo "${USER} ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers

# ------------------------------------------------------------
# Set-Up SSH with our Github deploy key
# ------------------------------------------------------------

ENV SSHDIR ${HOME}/.ssh/

RUN mkdir -p ${SSHDIR}

ADD ssh/config ${SSHDIR}/config
ADD ssh/id_rsa.mpi ${SSHDIR}/id_rsa
ADD ssh/id_rsa.mpi.pub ${SSHDIR}/id_rsa.pub
ADD ssh/id_rsa.mpi.pub ${SSHDIR}/authorized_keys

RUN chmod -R 600 ${SSHDIR}* && \
    chown -R ${USER}:${USER} ${SSHDIR}

RUN pip install --upgrade pip


USER ${USER}
RUN  pip install --user -U setuptools \
    && pip install --user mpi4py
    
  # Set default shell to /bin/bash
SHELL ["/bin/bash", "-cu"]  
   

# ------------------------------------------------------------
# Configure OpenMPI
# ------------------------------------------------------------

# Needed for string substitution
# SHELL ["/bin/bash", "-c"]
# Pick up some TF dependencies
# RUN apt-get update && apt-get install -y --no-install-recommends \
#         curl \
#         unzip


# ARG USE_PYTHON_3_NOT_2
# ARG _PY_SUFFIX=${USE_PYTHON_3_NOT_2:+3}
# ARG PYTHON=python${_PY_SUFFIX}
# ARG PIP=pip${_PY_SUFFIX}

# See http://bugs.python.org/issue19846
# ENV LANG C.UTF-8
# RUN apt-get update && apt-get install -y \
#     ${PYTHON} \
#     ${PYTHON}-pip

# RUN ${PIP} --no-cache-dir install --upgrade \
#     pip \
#     setuptools

# Some TF tools expect a "python" binary
# RUN ln -s $(which ${PYTHON}) /usr/local/bin/python 

# settings:
#   tensorflow
#   tensorflow-gpu
#   tf-nightly
#   tf-nightly-gpu
# Set --build-arg TF_PACKAGE_VERSION=1.11.0rc0 to install a specific version.
# Installs the latest version by default.
# ARG TF_PACKAGE=tensorflow
# ARG TF_PACKAGE_VERSION=
# RUN ${PIP} install ${TF_PACKAGE}${TF_PACKAGE_VERSION:+==${TF_PACKAGE_VERSION}}




# Install system packages



# Install conda
# ENV CONDA_DIR /opt/conda
# ENV PATH $CONDA_DIR/bin:$PATH

# RUN wget --quiet --no-check-certificate https://repo.continuum.io/miniconda/Miniconda3-4.2.12-Linux-x86_64.sh 

# RUN echo "c59b3dd3cad550ac7596e0d599b91e75d88826db132e4146030ef471bb434e9a *Miniconda3-4.2.12-Linux-x86_64.sh" | sha256sum -c - && \
#    /bin/bash /Miniconda3-4.2.12-Linux-x86_64.sh -f -b -p $CONDA_DIR 
# RUN  rm Miniconda3-4.2.12-Linux-x86_64.sh 
# RUN  echo export PATH=$CONDA_DIR/bin:'$PATH' > /etc/profile.d/conda.sh
USER root

RUN echo 'export PATH=/opt/conda/bin:$PATH' > /etc/profile.d/conda.sh && \
    wget --quiet https://repo.continuum.io/archive/Anaconda3-5.0.0.1-Linux-x86_64.sh -O ~/anaconda.sh && \
    /bin/bash ~/anaconda.sh -b -p /opt/conda && \
    rm ~/anaconda.sh

ENV PATH /opt/conda/bin:$PATH




# Install Python packages and keras
# ENV NB_USER keras
# ENV NB_UID 1000

# RUN useradd -m -s /bin/bash -N -u $NB_UID $NB_USER && \
#     chown $NB_USER $CONDA_DIR -R && \
#    mkdir -p /src && \
#    chown $NB_USER /src

# USER $NB_USER


# RUN rm -fr ${HOME}/.openmpi && mkdir -p ${HOME}/.openmpi
# ADD default-mca-params.conf ${HOME}/.openmpi/mca-params.conf
# RUN chown -R ${USER}:${USER} ${HOME}/.openmpi

ARG python_version=3.6

RUN conda config --append channels conda-forge

        
# Install git, wget, python-dev, pip, BLAS + LAPACK and other dependencies
# RUN apt-get update && apt-get install -y \
#  gfortran \
#  liblapack-dev \
#  libopenblas-dev \
#  python-dev \
#  python-tk\
#  git \
#  curl \
  # emacs24
  
      


    
ENV PATH /opt/conda/bin:$PATH
ENV PATH /opt/conda/envs/idp/bin:$PATH


RUN conda update conda

RUN conda config --add channels intel
RUN conda create -n idp intelpython3_full python=3
# RUN echo "source activate idp" > ~/.bashrc
# RUN echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc

# Install miniconda to /miniconda
# RUN curl -LO http://repo.continuum.io/miniconda/Miniconda-latest-Linux-x86_64.sh
# RUN bash Miniconda-latest-Linux-x86_64.sh -p /miniconda -b
# RUN rm /Miniconda-latest-Linux-x86_64.sh
# ENV PATH=/opt/conda/bin:${PATH}

# ENV PATH=/miniconda/envs/idp/bin:$PATH
# RUN conda remove -n tensorflow
# ARG python_version=3.6
## RUN pip install torch==${PYTORCH_VERSION} torchvision==${TORCHVISION_VERSION}
## RUN pip install mxnet==${MXNET_VERSION}
RUN conda config --append channels conda-forge
RUN conda install -y python=${python_version} && \
    # pip install --upgrade pip && \
    pip install \
      sklearn_pandas \
      h5py \
      MedPy \
      nibabel \
      Keras \
      numpy \
      scipy \
      Pillow \
      click \
      torch \
      mxnet \
      tensorflow && \
    conda install \
      bcolz \
      h5py \
      matplotlib \
      mkl \
      nose \
      notebook \
      pandas \
      pydot \
      pyyaml \
      scikit-learn \
      six \
      theano \
      mkdocs \
      && \
    git clone git://github.com/keras-team/keras.git /src && pip install -e /src[tests] && \
    pip install git+git://github.com/keras-team/keras.git && \
    conda clean -yt
    
# RUN python3 -m pip install --user pipx
# RUN python3 -m pipx ensurepath
# RUN python3 -m pip install -U pipx
# RUN pipx install python-dev-tools
# ENV TOOLS_PATH ls -l ~/.local/bin/whataformatter | sed -e "s/.*-> //" | sed -e "s#/bin.*#/bin#" TOOLS_PATH
# RUN pipx upgrade python-dev-tools   
# install CNN_GUI related packages
 # Install Horovod
# RUN HOROVOD_WITH_TENSORFLOW=1 HOROVOD_WITH_PYTORCH=1 HOROVOD_WITH_MXNET=1 \
  #  pip install --no-cache-dir horovod
RUN ln -s $(which python) /usr/local/bin/python
# Install OpenSSH for MPI to communicate between containers
RUN apt-get install -y --no-install-recommends openssh-client openssh-server && \
    mkdir -p /var/run/sshd

# Allow OpenSSH to talk to containers without asking for confirmation
RUN cat /etc/ssh/ssh_config | grep -v StrictHostKeyChecking > /etc/ssh/ssh_config.new && \
    echo "    StrictHostKeyChecking no" >> /etc/ssh/ssh_config.new && \
    mv /etc/ssh/ssh_config.new /etc/ssh/ssh_config


# ADD requirements.txt /requirements.txt
# RUN conda install numpy scipy mkl
# RUN conda install theano pygpu
# RUN pip install pip --upgrade
# RUN pip install -r /requirements.txt
# RUN pip uninstall protobuf
# RUN conda install tensorflow-gpu

# create a docker user
RUN useradd -ms /bin/bash docker
ENV HOME /home/docker

# copy necessary files to container
RUN mkdir $HOME/src
ENV PATH=/$HOME/src:${PATH}
ADD __init__.py $HOME/src/
ADD .theanorc $HOME/src/
# ADD .keras $HOME/src/
# RUN mkdir $HOME/src/.theanorc
# ENV PATH=/$HOME/src/.theanorc:${PATH}
# ADD .theanorc $HOME/src/.theanorc/
# RUN mkdir $HOME/src/.keras
# ENV PATH=/$HOME/src/.keras:${PATH}
# ADD .keras $HOME/src/.keras/
ADD app.py $HOME/src/
ADD CNN_GUI_scripts.py $HOME/src/
# ADD config $HOME/src/config
# ADD nets $HOME/src/nets
ADD libs $HOME/src/libs
ADD utils $HOME/src/utils
ADD ssh $HOME/src/ssh
ADD docker $HOME/src/docker
ADD logonic.png $HOME/src/
ADD nic_trainingwork_batch.py $HOME/src/
ADD nic_inference_batch.py $HOME/src/
ADD tensorboardlogs $HOME/src/
# add permissions (odd)
# RUN chown docker -R nets
# RUN chown docker -R config 

USER docker
WORKDIR $HOME/src
