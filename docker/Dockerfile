# This is an auto generated Dockerfile for ros:ros-base
# generated from docker_images/create_ros_image.Dockerfile.em
FROM ros:noetic-ros-core-focal

# Ensure bash and tmux are installed
USER root
RUN mkdir -p /var/lib/apt/lists/partial
RUN chmod a+rwx /var/lib/apt/lists/partial

# Install necessary packages including bash and tmux
RUN apt-get update && apt-get install -y bash tmux

# Create a non-root user if it does not exist and give sudo privileges
RUN id -u docker &>/dev/null || useradd -m -s /bin/bash docker && echo "docker:docker" | chpasswd && adduser docker sudo

# Set Bash as the default shell
SHELL ["/bin/bash", "-c"]

# Set the home directory for docker user
ENV HOME /home/docker 

# install bootstrap tools
RUN apt-get update && apt-get install --no-install-recommends -y \
    build-essential \
    python3-rosdep \
    python3-rosinstall \
    python3-vcstools \
    && rm -rf /var/lib/apt/lists/*

# bootstrap rosdep
RUN if [ ! -f /etc/ros/rosdep/sources.list.d/20-default.list ]; then \
        rosdep init; \
    fi
RUN rosdep update --rosdistro noetic

# install ros packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    ros-noetic-ros-base=1.5.0-1* \
    && rm -rf /var/lib/apt/lists/*

# Install pip and Python dependencies
RUN apt-get update && apt-get install -y \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

#add dependencies for LLM Guided seeding
COPY ./requirements.txt /app/
RUN pip install -r /app/requirements.txt

# Switch to your new user
USER docker
