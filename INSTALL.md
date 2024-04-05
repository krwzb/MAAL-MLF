## Installation

Most of the requirements of this projects are exactly the same as [maskrcnn-benchmark](https://github.com/facebookresearch/maskrcnn-benchmark). If you have any problem of your environment, you should check their [issues page](https://github.com/facebookresearch/maskrcnn-benchmark/issues) first. Hope you will find the answer.

### Requirements:
- Python <= 3.8
- PyTorch >= 1.2 (Mine 1.4.0 (CUDA 10.1))
- torchvision >= 0.4 (Mine 0.5.0 (CUDA 10.1))
- cocoapi
- yacs
- matplotlib
- GCC >= 4.9
- OpenCV


### Step-by-step installation

```bash
# first, make sure that your conda is setup properly with the right environment
# for that, check that `which conda`, `which pip` and `which python` points to the
# right path. From a clean conda env, this is what you need to do

pip install torch==1.7.1+cu110 torchvision==0.8.2+cu110 torchaudio==0.7.2 -f https://download.pytorch.org/whl/cu110/torch_stable.html

conda create --name MSPN
conda activate MSPN

# this installs the right pip and dependencies for the fresh python
conda install ipython scipy h5py -y

# scene_graph_benchmark and coco api dependencies
pip install ninja yacs cython matplotlib tqdm opencv-python overrides

#install PyCOCO tools (cocoapi)
git clone https://github.com/cocodataset/cocoapi.git
cd cocoapi/PythonAPI ; python setup.py build_ext install

#install apex
git clone --branch 22.04-dev https://github.com/NVIDIA/apex.git
cd apex ; python setup.py install


python setup.py build develop

```