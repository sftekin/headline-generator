FROM pytorch/pytorch:1.4-cuda10.1-cudnn7-runtime

ENV DEBIAN_FRONTEND=noninteractive

ENV LC_ALL C.UTF-8
ENV LANG C.UTF-8

RUN pip install matplotlib==3.2.1
RUN pip install pandas==1.0.3
RUN pip install easybert==1.0.3
RUN pip install nltk==3.5
RUN pip install sumy==0.8.1

RUN python -c "import nltk; nltk.download('punkt')"
RUN python -c "import nltk; nltk.download('stopwords')"

WORKDIR /workspace
