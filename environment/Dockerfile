# hash:sha256:e0b8d7ab5041e34ce6cfc10fbf8ef1b30648fd9f8d3d59cdce2d2253e96c160a
ARG REGISTRY_HOST
FROM $REGISTRY_HOST/codeocean/jupyterlab:3.0.9-miniconda4.9.2-python3.8-ubuntu20.04

ARG DEBIAN_FRONTEND=noninteractive

ARG GIT_ASKPASS
ARG GIT_ACCESS_TOKEN
COPY git-askpass /

ADD "https://github.com/coder/code-server/releases/download/v4.9.0/code-server-4.9.0-linux-amd64.tar.gz" /.code-server/code-server.tar.gz

RUN pip3 install -U --no-cache-dir \
	dask \
	zarr \
    matplotlib \
	pandas \
	scikit-learn \
	seaborn \
	scipy \ 
	aind-codeocean-api \
    aind-codeocean-utils \
    pyyaml \
	dask_ml \
	pytest

RUN cd /.code-server \
	&& tar -xvf code-server.tar.gz \
	&& rm code-server.tar.gz \
	&& ln -s /.code-server/code-server-4.9.0-linux-amd64/bin/code-server  /usr/bin/code-server
