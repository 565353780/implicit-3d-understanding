sudo apt install xvfb ninja-build freeglut3-dev libglew-dev meshlab
sudo apt install mesa-common-dev libglu1-mesa-dev libosmesa6-dev libxi-dev libgl1-mesa-dev
sudo apt install --reinstall libgl1-mesa-glx

conda create -n im3d python=3.8
conda activate im3d
pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113
pip install wandb tqdm cython

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

