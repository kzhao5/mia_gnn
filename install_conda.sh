#!/bin/bash
#SBATCH --job-name=install_conda
#SBATCH --output=install_conda.log
#SBATCH --error=install_conda.err
#SBATCH --time=00:30:00
#SBATCH --mem=4G

# 下载Miniconda安装脚本
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh

# 安装Miniconda
bash miniconda.sh -b -p $HOME/miniconda

# 初始化conda
eval "$($HOME/miniconda/bin/conda shell.bash hook)"

# 配置conda
conda config --set auto_activate_base false
conda init

echo "Conda installation complete. Please log out and log back in, or run 'source ~/.bashrc' to use conda."