FROM ubuntu:20.04
ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update && \
    apt-get install -y \
    python3 \
    python3-pip \
    ffmpeg \
    libsm6 \
    libxext6 \
    git
RUN git clone https://github.com/elenaliao1002/hakka App
WORKDIR /App/website
RUN pip install -r requirements.txt
CMD [ "streamlit", "run", "website/app.py" ]