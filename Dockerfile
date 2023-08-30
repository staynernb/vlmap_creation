FROM python:3.9
WORKDIR /code

COPY ./requirements.txt /code/requirements.txt

RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y

RUN --mount=type=cache,target=/root/.cache/pip pip install -r /code/requirements.txt

RUN useradd -m -u 1000 user
#USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH
	
RUN mkdir /app

WORKDIR /app

RUN git clone https://github.com/vlmaps/vlmaps

RUN bash
	



