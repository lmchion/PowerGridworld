
name="John Doe"
email="johndoe@example.com"

# install git
sudo yum update 
sudo dnf upgrade --releasever=2023.0.20230322
sudo yum install git -y
sudo yum install python3-pip
sudo yum install docker
sudo usermod -a -G docker ec2-user
sudo systemctl enable docker.service
sudo systemctl start docker.service
git config --global user.name ${name}
git config --global user.name ${email}


# install aws-cli https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html
mkdir installers
cd installers
curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
unzip awscliv2.zip
sudo ./aws/install
sudo yum install -y amazon-linux-extras

# install AMD drivers https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/install-amd-driver.html
wget https://dl.fedoraproject.org/pub/epel/epel-release-latest-8.noarch.rpm   #install epel
sudo rpm -ihv --nodeps ./epel-release-latest-8.noarch.rpm
aws s3 cp --recursive s3://ec2-amd-linux-drivers/latest/ .
tar -xf amdgpu-pro-*rhel*.tar.xz
cd amdgpu-pro-20.20-1184451-rhel-7.8
./amdgpu-pro-install -y --opencl=pal,legacy

# install miniconda https://docs.conda.io/en/latest/miniconda.html
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