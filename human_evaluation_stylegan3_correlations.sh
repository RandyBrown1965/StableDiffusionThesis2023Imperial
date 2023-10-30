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

image_dirs='images_out_BatchA_renumbered images_out_BatchB_renumbered'
models='stylegan3-r-ffhq-1024x1024 stylegan3-r-ffhqu-256x256 stylegan3-t-ffhqu-256x256 stylegan3-r-afhqv2-512x512 stylegan2-ffhq-512x512 stylegan2-cifar10-32x32'

for image_dir in $image_dirs
do
	for model in $models
	do
		echo "python human_evaluation_stylegan3_correlations.py $image_dir --model $model"
		python human_evaluation_stylegan3_correlations.py $image_dir --model $model
	done
done

TERM=vt100 # or TERM=xterm
/usr/bin/nvidia-smi
pip list
uptime
