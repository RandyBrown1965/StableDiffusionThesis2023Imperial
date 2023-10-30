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


"""
python proposed_inpainting_pipeline.py --prompt "Beautiful brunette wielding a flaming sword, battle field background, all in focus" --solid_mask --generator sd-v1.5 sd-v2.1 sdxl-base sdxl-refiner


python proposed_inpainting_pipeline.py --prompt "Beautiful brunette wielding a flaming sword, battle field background, all in focus" --solid_mask --inpainter sd-v1 sd-v2 sdxl-base sdxl-refiner


python proposed_inpainting_pipeline.py --prompt "Beautiful brunette wielding a flaming sword, battle field background, all in focus" --solid_mask --generator sd-v1.5 sdxl-base  --inpainter sd-v2 sdxl-base
"""

python proposed_inpainting_pipeline.py --prompt "Beautiful brunette wielding a flaming sword, battle field background, all in focus" --solid_mask --generator sd-v1.5 sd-v2.1 sdxl-base sdxl-refiner --inpainter sd-v1 sd-v2 sdxl-base sdxl-refiner --random_seed 222



TERM=vt100 # or TERM=xterm
/usr/bin/nvidia-smi
pip list
uptime
