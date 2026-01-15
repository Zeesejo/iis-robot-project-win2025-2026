# 1. Use the official Jupyter 3.10 image
FROM jupyter/base-notebook:python-3.10

USER root

# 2. Install system dependencies (Logic + Graphics)
# swi-prolog: for knowledge_reasoning.py
# mesa/opengl/xvfb: for PyBullet rendering in src/environment
RUN apt-get update && apt-get install -y \
    swi-prolog \
    libgl1-mesa-glx \
    libgl1-mesa-dev \
    python3-opengl \
    xvfb \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

USER ${NB_UID}

# 3. Handle Python Requirements
# Assuming you run: docker build -t robot-proj . from the project root
COPY --chown=${NB_UID}:${NB_GID} requirements.txt /tmp/requirements.txt

# Install libraries (pyswip, pybullet, numpy, etc.)
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# 4. Set the Working Directory 
WORKDIR /home/jovyan/work

# 5. Copy the rest of your project structure
# This will copy 'src', 'executables', 'imgs', etc. into the work folder
COPY --chown=${NB_UID}:${NB_GID} . /home/jovyan/work

# 6. Set Environment Variables for Graphics
ENV DISPLAY=:99
