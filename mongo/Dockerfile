FROM harbor.mobagel.com/core-tech-release/mongo:4.2

RUN mkdir /home/mongodb && \
        echo "" > /home/mongodb/.dbshell && \
        chown -R mongodb:mongodb /home/mongodb/.dbshell

COPY ./mongo-startup.sh /docker-entrypoint-initdb.d/init-mongo.sh

EXPOSE 27017
