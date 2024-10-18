#!/bin/bash
# SBATCH -J herewegoagain
# SBATCH -p high
# SBATCH -N 1
# SBATCH --nodelist=node032
# SBATCH --chdir=/home/jreyes
# SBATCH --mem=32GB 
# SBATCH --gres=gpu:gtx1080:1

#SBATCH -o /home/ytanriverdi/%J.%u.out # STDOUT
#SBATCH -e /home/ytanriverdi/%J.%u.err # STDERR 



## SBATCH --partition=<partition>          # Partition/queue name
## SBATCH --nodes=<num_nodes>              # Number of nodes to use
## SBATCH --ntasks=<num_tasks>             # Number of tasks (can differ from nodes)
## SBATCH --cpus-per-task=<num_cpus>       # Number of CPUs per task
## SBATCH --gres=gpu:<type>:<num_gpus>     # Request GPUs (type and number)
## SBATCH --time=<time>                    # Maximum job time (format D-HH:MM:SS)
## SBATCH --mem=<memory>                   # Memory per node
## SBATCH --mail-user=<email>              # Email for notifications
## SBATCH --mail-type=BEGIN,END,FAIL       # Notification types: start, end, fail
## SBATCH --output=<path_to_stdout>        # Path for standard output (STDOUT)
## SBATCH --error=<path_to_stderr>         # Path for error output (STDERR)
## SBATCH --nodelist=<node_list>           # Specific nodes to use
## SBATCH --exclude=<node_list>            # Nodes to exclude
## SBATCH --chdir=<directory>              # Working directory
## SBATCH --dependency=<job_id>            # Start after a specific job
## SBATCH --requeue                        # Requeue job if it fails
## SBATCH --array=<index_list>             # Run an array of jobs
## SBATCH --account=<account>              # Billing account (if required)
## SBATCH --constraint=<features>          # Restrict to nodes with specific features

# Load CUDA module
module load CUDA/12.1

# Load Anaconda module
module load Anaconda3/2020.02

# Update Conda and create environment if needed
if ! conda info --envs | grep -q herewegoagain; then
    echo "Creating conda environment 'herewegoagain'.........................."
    conda create -n herewegoagain python=3.10.13 -y
else
    echo "Conda environment 'herewegoagain' already exists!"
fi

source activate herewegoagain

# Install necessary libraries
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu121


echo "Environment is ready!" 

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
