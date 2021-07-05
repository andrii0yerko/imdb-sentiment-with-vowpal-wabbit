FROM python:3.8

ARG vw_branch_or_tag=8.10.2

# install vowpal wabbit
RUN apt-get update && apt-get install cmake libboost-all-dev -y && \
    git clone -b $vw_branch_or_tag --single-branch --depth=1 --recursive https://github.com/VowpalWabbit/vowpal_wabbit.git /vw && \
    cd /vw && \
    make && \
    make install && \
    cd / && \
    rm -rf /vw && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt ./
RUN pip3 install -r requirements.txt && python -m nltk.downloader stopwords

RUN export PYTHONPATH='${PYTHONPATH}:/app'

COPY . .

CMD ["python", "./run.py"]