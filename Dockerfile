FROM continuumio/anaconda3

RUN conda install -c pytorch pytorch
RUN conda install -c conda-forge streamlit
RUN conda install -c anaconda pillow 
RUN conda install -c anaconda pandas
RUN conda install -c conda-forge opencv
RUN conda install -c simpleitk simpleitk
RUN conda install -c conda-forge h5py
RUN conda install -c anaconda pyyaml
RUN conda install -c anaconda numpy
RUN pip install adabelief-pytorch
RUN conda install -c conda-forge pyvista
RUN conda install -c plotly plotly
RUN conda install -c conda-forge matplotlib
RUN conda install -c pytorch torchvision
RUN conda install -c conda-forge tensorboard
RUN conda install -c conda-forge tqdm
RUN pip install unet
RUN pip install pytorch-wavelet
RUN conda install -c conda-forge albumentations
RUN conda install -c anaconda scikit-learn
RUN conda install -c conda-forge vtk
RUN pip install streamlit-server-state
RUN conda install pytorch torchvision cudatoolkit=10.2 -c pytorch
RUN conda clean -afy

COPY RDN_segmentation_container .
