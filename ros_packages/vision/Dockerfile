FROM luxonis/depthai-ros:humble-arm64-latest
SHELL ["/bin/bash", "-c"]

WORKDIR /app
COPY ./datatypes ros2_ws/datatypes
COPY ./vision ros2_ws/vision

RUN set -xe \
    && apt-get update \
    && apt-get install -y python3-pip

RUN pip install depthai==2.25.1.0

RUN cd ros2_ws && \
    source /opt/ros/humble/setup.bash && \
    colcon build

COPY ./ros_entrypoint.sh /

RUN chmod +x /ros_entrypoint.sh

ENTRYPOINT ["/ros_entrypoint.sh"]
CMD ["bash"]
