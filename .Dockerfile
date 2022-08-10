FROM ubuntu:20.04

WORKDIR /root/projects

RUN apt update && apt upgrade -y
RUN apt install python3 python3-pip -y

# RUN pip3 install numpy ipykernel torch sklearn
CMD tail -f /dev/null