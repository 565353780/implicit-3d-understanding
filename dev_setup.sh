sudo apt install xvfb ninja-build freeglut3-dev libglew-dev meshlab -y
sudo apt install mesa-common-dev libglu1-mesa-dev libosmesa6-dev libxi-dev libgl1-mesa-dev -y
sudo apt install --reinstall libgl1-mesa-glx -y
sudo apt-get install openmpi-bin openmpi-common openmpi-doc libopenmpi-dev

pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 torchaudio==0.13.1 \
  --extra-index-url https://download.pytorch.org/whl/cu116

pip install wandb tqdm cython pytz python-dateutil \
      trimesh scipy scikit-image shapely jellyfish \
      vtk seaborn h5py opencv-python tensorboard

cd external/ldif/gaps
make mesa -j

cd ../../mesh_fusion/libfusiongpu
mkdir build
cd build
cmake ..
make -j
cd ..
python setup.py build_ext -i -f
cd ../librender
python setup.py build_ext -i -f
cd ../libmcubes
python setup.py build_ext -i -f

cd ../../ldif/ldif2mesh
bash build.sh

cd ../../..

