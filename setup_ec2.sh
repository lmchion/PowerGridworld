
name="Luis Chion"
email="lmchion@berkeley.edu"

# install git
sudo yum update 
#sudo dnf upgrade --releasever=2023.0.20230322
sudo yum install git -y
sudo yum install python3-pip
sudo yum install docker
sudo usermod -a -G docker ec2-user
sudo systemctl enable docker.service
sudo systemctl start docker.service
git config --global user.name ${name}
git config --global user.email ${email}


# install aws-cli https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html
cd ~
mkdir installers
cd installers
curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
unzip awscliv2.zip
sudo ./aws/install


# install miniconda https://docs.conda.io/en/latest/miniconda.html
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh

# vs code extensions
code --install-extension mhutchie.git-graph
code --install-extension ms-azuretools.vscode-docker
code --install-extension ms-python.python

# pull repo
cd ~
#git clone https://github.com/lmchion/PowerGridworld.git
cd PowerGridworld
conda create -n gridworld_hs python=3.10 -y
conda activate gridworld_hs
pip install -e .
pip install -r requirements_hs_new.txt

mkdir data
mkdir data/inputs
mkdir data/outputs
mkdir data/outputs/ray_results
mkdir data/outputs/ray_results/PPO
