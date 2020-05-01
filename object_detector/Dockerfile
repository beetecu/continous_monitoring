FROM openvinodevosdk/basic
LABEL maintainer="eliza.gonzalez@devo.com"

RUN mkdir APP


RUN mkdir APP/CERTS
COPY CERTS APP/CERTS

RUN mkdir APP/models
COPY models APP/models

RUN mkdir APP/utils
COPY code/utils APP/utils


COPY code/*.py APP/

WORKDIR APP

ADD start.sh /
ENTRYPOINT ["/start.sh"]


