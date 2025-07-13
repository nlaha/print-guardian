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

# Add healthcheck that waits for the .ready file to be created
HEALTHCHECK --interval=10s --timeout=3s --start-period=30s --retries=3 \
	CMD test -f .ready || exit 1

# Clean up any existing .ready file on startup and then run the application
CMD ["sh", "-c", "rm -f .ready && print-guardian"]
