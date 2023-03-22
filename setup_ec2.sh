# install git
sudo yum update 
sudo yum install git -y
sudo yum install python3-pip
sudo yum install docker
sudo usermod -a -G docker ec2-user
sudo systemctl enable docker.service
sudo systemctl start docker.service

# install miniconda https://docs.conda.io/en/latest/miniconda.html
mkdir installers
cd installers
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh

# vs code extensions
code --install-extension mhutchie.git-graph
code --install-extension ms-azuretools.vscode-docker
code --install-extension ms-python.python

# pull repo
cd ~
git clone https://github.com/lmchion/PowerGridworld.git
cd PowerGridworld