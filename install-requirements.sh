conda create -n rl_examples python=3.8 -y
conda activate rl_examples

# CPU install
pip3 install torch torchvision torchaudio --proxy=http://127.0.0.1:7890

# GPU cuda11.08 install
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

pip3 install -r requirements.txt --proxy=http://127.0.0.1:7890
