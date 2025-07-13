FROM rust:1

WORKDIR /usr/src/app
COPY . .

RUN apt-get update && \
    apt-get install -y cmake build-essential libssl-dev pkg-config

RUN cargo install --path .

CMD ["print-guardian"]