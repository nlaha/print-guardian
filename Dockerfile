FROM rust:1

WORKDIR /usr/src/app
COPY . .

RUN apt-get update && apt-get install -y \ 
	curl \ 
	git \ 
	clang \ 
	libclang-dev \ 
	build-essential \ 
	cmake \ 
	libcurl4-openssl-dev \ 
	libssl-dev \ 
	make

RUN cargo install --path .

CMD ["print-guardian"]
