# A Comparative Study on State-Action Spaces for Learning Viewpoint Selection and Manipulation with Diffusion Policy

## Tested Hardware
- AMD PRO 5975WX CPU
- Dual 4090
- 128GB RAM

## Setup
1. Clone the repository and its submodules
2. Install the dependencies of the gym
   ```
    git submodule update --init --recursive
    cd mavis_mujoco
    mamba env create -f environment.yml 
    conda activate mavis-mujoco
    cd xArm-Python-SDK
    python setup.py install
    cd ..
    pip install -e .
    ```
3. Install other packages
   ```
    cd ..
    pip install -r requirements.txt
    ```

## Usage
- To train diffusion policy for the pick-and-place scenario:
   ```
   python train_diffusion_policy.py
   ```
- To perform inference and evaluate all ckeckpoints in a training session:
   ```
   python inference_and_eval.py
   ```