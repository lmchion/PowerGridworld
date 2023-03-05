
######  BUILD ###########

FROM python:3.8-slim as builder

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

RUN ls
RUN echo "$PWD"

RUN python -m pip install --upgrade pip
RUN pip install -r /PowerGridworld/requirements.txt --default-timeout=1000 --no-cache-dir
RUN pip install -r /PowerGridworld/examples/marl/rllib/requirements.txt --default-timeout=1000 --no-cache-dir

# # copy poetry package definitions
# COPY pyproject.toml poetry.lock ./

# # copy the environment variable from local to docker
# RUN python -m venv --copies ./venv

# # activate virtual environment and install poetry but only main
# RUN . ./venv/bin/activate && poetry install --only main




######  RUNNER ###########


# FROM python:3.8-slim as runner

# RUN set -eux; \
#     apt-get update; \
#     apt-get upgrade -y; \
#     apt-get install -y curl build-essential libffi-dev ;  \
#     rm -rf /var/lib/apt/lists/*

# COPY --from=builder /venv /venv
# ENV PATH /venv/bin:$PATH

# COPY . ./

# CMD ["uvicorn","src.main:app","--port","8000","--host","0.0.0.0"]

# HEALTHCHECK --start-period=10s  --retries=4  CMD curl --fail http://localhost:8000/health || exit 1

#HEALTHCHECK --start-period=10s  --retries=4  CMD python -c "import requests; requests.get('http://localhost:8000/health', timeout=4)" || exit 1