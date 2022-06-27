sudo apt install xvfb ninja-build freeglut3-dev libglew-dev meshlab
sudo apt install mesa-common-dev libglu1-mesa-dev libosmesa6-dev libxi-dev libgl1-mesa-dev
sudo apt install --reinstall libgl1-mesa-glx

conda create -n im3d python=3.8
conda activate im3d
pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113
pip install wandb tqdm cython pytz python-dateutil trimesh scipy scikit-image shapely jellyfish vtk seaborn h5py

./build.sh

