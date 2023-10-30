#!/bin/bash
##SBATCH --partition=gpgpuB
#SBATCH --gres=gpu:1
#SBATCH --mail-type=ALL # required to send email notifcations
#SBATCH --mail-user=rdb121 # required to send email notifcations - please replace <your_username> with your college login name or email address
#export PATH=/vol/bitbucket/${USER}/myvenv/bin/:$PATH # removed for conda venv

echo "source ~/.bashrc"
source ~/.bashrc

# PIP ENV
echo "source /vol/bitbucket/rdb121/venv/sdxlstylegan3venv_pip2/bin/activate"
source /vol/bitbucket/rdb121/venv/sdxlstylegan3venv_pip2/bin/activate


python human_evaluation_pickapic_correlations.py images_out_BatchB_renumbered --image_ref SDXL_astronaut_jungle_refined_photo.png
python human_evaluation_pickapic_correlations.py images_out_BatchB_renumbered --image_ref PhotoOfABrunetteHoldingABookAvoidMangledFingers04.jpg
python human_evaluation_pickapic_correlations.py images_out_BatchB_renumbered --image_ref queens_tower.jpg
python human_evaluation_pickapic_correlations.py images_out_BatchB_renumbered --image_ref MSc_bowling_28June.jpg
python human_evaluation_pickapic_correlations.py images_out_BatchB_renumbered --image_ref www.Panayispicture.com-332.jpg

TERM=vt100 # or TERM=xterm
/usr/bin/nvidia-smi
pip list
uptime
