#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --mail-type=ALL # required to send email notifcations
#SBATCH --mail-user=rdb121 # required to send email notifcations - please replace <your_username> with your college login name or email address
#export PATH=/vol/bitbucket/${USER}/myvenv/bin/:$PATH # removed for conda venv
#source activate    # removed for conda venv
echo "source ~/.bashrc"
source ~/.bashrc
#source /vol/cuda/11.0.3-cudnn8.0.5.39/setup.sh # This is from the Imperial website
#echo "source /vol/cuda/11.4.120-cudnn8.2.4/setup.sh # This is from Marta, removed for condavenv"
#source /vol/cuda/11.4.120-cudnn8.2.4/setup.sh # This is from Marta, removed for condavenv
#export NEPTUNE_API_TOKEN=<your-token>

echo "conda init bash"
conda init bash
echo "source /vol/bitbucket/rdb121/miniconda3/bin/activate"
source /vol/bitbucket/rdb121/miniconda3/bin/activate
echo "conda activate sdxlstylegan3venv"
conda activate sdxlstylegan3venv

python ~/Thesis/sdxl_refiner_ensemble.py


TERM=vt100 # or TERM=xterm
/usr/bin/nvidia-smi
conda list
uptime
