FROM ros:jazzy-ros-core

# Install dependencies
RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-colcon-common-extensions

# Create a workspace directory
RUN mkdir -p /root/ros2_ws/src

# Copy the workspace to the Docker image
COPY . /root/ros2_ws

# Build the workspace
RUN /bin/bash -c "source /opt/ros/jazzy/setup.bash && cd /root/ros2_ws && colcon build"

# Ensure the ROS 2 setup files are sourced in every new shell session
RUN echo "source /opt/ros/jazzy/setup.bash" >> /root/.bashrc
RUN echo "source /root/ros2_ws/install/local_setup.bash" >> /root/.bashrc

# Set the entrypoint
# possibly change entry point aswell "source /root/.bashrc && exec bash && ros2 launch my_test_package publisher.launch.py"
ENTRYPOINT ["/bin/bash", "-c", "source /root/.bashrc && exec bash"]
