conda create -n rl_examples python=3.8 -y
conda activate rl_examples

pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip3 install -r requirements.txt
