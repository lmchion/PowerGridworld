
######  BUILD ###########

FROM python:3.8-slim as builder

# set working directory
#WORKDIR /app

# Turns off writing .pyc files
ENV PYTHONDONTWRITEBYTECODE=1

# Seems to speed things up
ENV PYTHONUNBUFFERED=1

# add gridworld env to path
ENV PATH /venv/bin:$PATH

RUN set -eux; \    
    apt-get update; \
    apt-get upgrade -y; \
    apt-get install -y build-essential libffi-dev ;  \
    rm -rf /var/lib/apt/lists/*

# set -eux -> helps you to see in the logs where a failure occurred. It writes the commands line by 
# line while it’s running, and stops the script right away when there is a failure.
# apt-get update -> fetches the latest version of the package list from your distro's software repository, 
# and any third-party repositories you may have configured
# apt-get upgrade -> ommand downloads and installs the updates for each outdated package and dependency on your system

COPY . /PowerGridworld/

RUN python3 -m pip install --upgrade pip


#RUN conda create -n gridworld python=3.8 -y
#RUN conda activate gridworld

RUN python3 -m venv --copies ./venv

RUN . ./venv/bin/activate

RUN pip install -e /PowerGridworld
RUN pip install -r /PowerGridworld/requirements.txt --default-timeout=1000 --no-cache-dir
RUN pip install -r /PowerGridworld/examples/marl/rllib/requirements.txt --default-timeout=1000 --no-cache-dir

# # copy poetry package definitions
# COPY pyproject.toml poetry.lock ./

# # copy the environment variable from local to docker
# RUN python -m venv --copies ./venv

# # activate virtual environment and install poetry but only main
# RUN . ./venv/bin/activate && poetry install --only main




######  RUNNER ###########


FROM python:3.8-slim as runner

# Extra python env
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_DISABLE_PIP_VERSION_CHECK=1
ENV PATH="/venv/bin:$PATH"

#WORKDIR /app

RUN set -eux; \
     apt-get update; \
     apt-get upgrade -y; \
     apt-get install -y curl build-essential libffi-dev ;  \
     rm -rf /var/lib/apt/lists/*

COPY --from=builder /venv /venv

COPY . /PowerGridworld

#CMD ["uvicorn","src.main:app","--port","8000","--host","0.0.0.0"]
#RUN python /PowerGridworld/examples/marl/rllib/heterogeneous/train_hs.py 
