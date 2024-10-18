#!/bin/bash
# SBATCH -J segment3d
# SBATCH -p high
# SBATCH -N 1
# SBATCH --nodelist=node032
# SBATCH --chdir=/home/ytanriverdi/
# SBATCH --mem=32GB 
# SBATCH --gres=gpu:gtx1080:1

#SBATCH -o /home/ytanriverdi/segment3d/log/out/loaders%J.%u.out # STDOUT
#SBATCH -e /home/ytanriverdi/segment3d/log/err/loaders%J.%u.err # STDERR 

# Load CUDA module
module load CUDA/12.1

nvcc --version

# Load Anaconda module
module load Anaconda3/2020.02

# Update if necessary.
# conda update -n base -c defaults conda # Permission denied.

# Check if the Conda environment 'segment3d' already exists
if ! conda info --envs | grep -q segment3d; then
    echo "Creating conda environment 'segment3d'.........................."
    conda create -n segment3d python=3.10.13 -y
else
    echo "Conda environment 'segment3d' already exists!"
fi

source activate segment3d

# Check requirements.
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install --user -r requirements.txt

echo "Environment is here!" 

#!/bin/bash

# Get Python version
python_version=$(python --version 2>&1)
echo "Python version: $python_version"

# Check CUDA availability using Python
cuda_status=$(python - << EOF
import torch
print(torch.cuda.is_available())
EOF
)

# Echo CUDA status
if [ "$cuda_status" = "True" ]; then
    echo "CUDA is available"
    CUDA_AVAILABLE=true
else
    echo "CUDA is not available"
    CUDA_AVAILABLE=false
fi

# Export CUDA status as an environment variable
export CUDA_AVAILABLE
echo "CUDA_AVAILABLE=$CUDA_AVAILABLE"


# python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'

# python3 ./scripts/main.py --ims_per_batch 32 --batch_size_per_image 128 --num_workers 2 --unfreeze_backbone True --freeze_at_block 3 --output_dir "./output_pw"

python loaders.py