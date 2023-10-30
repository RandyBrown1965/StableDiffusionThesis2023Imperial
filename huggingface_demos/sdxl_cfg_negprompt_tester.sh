#!/bin/bash
##SBATCH --partition=gpgpuB
#SBATCH --gres=gpu:1
#SBATCH --mail-type=ALL # required to send email notifcations
#SBATCH --mail-user=rdb121 # required to send email notifcations - please replace <your_username> with your college login name or email address
#export PATH=/vol/bitbucket/${USER}/myvenv/bin/:$PATH # removed for conda venv

echo "source ~/.bashrc"
source ~/.bashrc

# CONDA ENV
#echo "conda init bash"
#conda init bash
#echo "source /vol/bitbucket/rdb121/miniconda3/bin/activate"
#source /vol/bitbucket/rdb121/miniconda3/bin/activate
#echo "conda activate sdxlstylegan3venv_save3"
#conda activate sdxlstylegan3venv_save3

# PIP ENV
echo "source /vol/bitbucket/rdb121/venv/sdxlstylegan3venv_pip/bin/activate"
source /vol/bitbucket/rdb121/venv/sdxlstylegan3venv_pip/bin/activate

echo "python ~/Thesis/sdxl_cfg_negprompt_tester.py"
python ~/Thesis/sdxl_cfg_negprompt_tester.py
#source /vol/cuda/11.0.3-cudnn8.0.5.39/setup.sh # This is from the Imperial website
#echo "source /vol/cuda/11.4.120-cudnn8.2.4/setup.sh # This is from Marta, removed for condavenv"
#source /vol/cuda/11.4.120-cudnn8.2.4/setup.sh # This is from Marta, removed for condavenv
#export NEPTUNE_API_TOKEN=<your-token>

TERM=vt100 # or TERM=xterm
/usr/bin/nvidia-smi
pip list
uptime
