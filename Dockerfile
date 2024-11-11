FROM ufoym/deepo:latest 

RUN useradd --uid 8877 --create-home nonroot

RUN usermod -a -G sudo nonroot

RUN echo 'nonroot:hiaac' | chpasswd

RUN chown nonroot /home/nonroot/

WORKDIR /home/nonroot/

COPY . /home/nonroot/

RUN pip install seaborn

RUN pip install keras-tuner

RUN pip install tensorflow-hub

RUN pip install tf-models-official

RUN pip install numpy

RUN pip install transformers

RUN pip install -r requirements.txt

RUN chown -R $USER:$USER .
