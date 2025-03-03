pip install requests
pip install tqdm
sudo apt update
git lfs install
sudo apt install ffmpeg --yes
sudo apt update --yes
sudo apt install ninja-build --yes
sudo apt install zram-config --yes
df -h /dev/shm

git clone https://github.com/FurkanGozukara/Wan2.1

cd Wan2.1

python3 -m venv venv

source ./venv/bin/activate

echo "Installing requirements"

pip install torch==2.5.1 torchvision --index-url https://download.pytorch.org/whl/cu124

pip install deepspeed

pip install xformers==0.0.28.post3 --index-url https://download.pytorch.org/whl/cu124

pip install triton

pip install wheel

pip install torchao

pip install -r requirements.txt

pip install huggingface_hub ipywidgets hf_transfer

pip install easydict

pip install moviepy

git clone https://github.com/modelscope/DiffSynth-Studio
cd DiffSynth-Studio
pip install -e .

cd ..

git clone https://github.com/FurkanGozukara/Practical-RIFE
cd Practical-RIFE
pip3 install -r requirements.txt

cd ..

cd ..

python Download_RIFE.py


echo "Virtual environment made and installed properly"

# Keep the terminal open
read -p "Press Enter to continue..."

