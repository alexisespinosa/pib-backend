FROM luxonis/depthai-ros:humble-arm64-latest
SHELL ["/bin/bash", "-c"]

WORKDIR /app
COPY ./ros_packages/datatypes ros2_ws/datatypes

RUN set -xe \
    && apt-get update \
    && apt-get install -y python3-pip

RUN pip install depthai==2.25.1.0
RUN pip install blobconverter==1.4.2

COPY ./pib_api/client /tmp/pib_api_client
RUN pip install /tmp/pib_api_client

# Pre-bake NN models into the image so runtime never requires network
# access to Luxonis. blobconverter.from_zoo() invokes their online
# compilation service (model -> MyriadX .blob) and caches the result in
# /root/.cache/blobconverter/. Located before the ./vision COPY so
# source-only changes don't invalidate this layer.
# See docs/vision-architecture.md §4.5.
RUN python3 -c "import blobconverter; blobconverter.from_zoo(name='face-detection-retail-0004', shaves=6)"
RUN python3 -c "import blobconverter; blobconverter.from_zoo(name='face-recognition-arcface-112x112', zoo_type='depthai', shaves=6)"

COPY ./ros_packages/vision ros2_ws/vision

RUN cd ros2_ws && \
    source /opt/ros/humble/setup.bash && \
    colcon build

COPY ./ros_packages/ros_entrypoint.sh /

RUN chmod +x /ros_entrypoint.sh

ENTRYPOINT ["/ros_entrypoint.sh"]
CMD ["bash"]
