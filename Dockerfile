FROM debian:bookworm

ENV DEBIAN_FRONTEND=noninteractive
SHELL ["/bin/bash", "-o", "pipefail", "-c"]
WORKDIR /app
# Update packages
RUN apt-get update && apt-get upgrade --no-install-recommends -y && apt-get install pip make wget -y --no-install-recommends
# Pull in data
COPY ./iso_forest ./iso_forest
# Install dependencies
RUN pip install --break-system-packages --upgrade pip && pip install --break-system-packages -r /app/iso_forest/requirements.txt 
# Install Go
COPY ./Makefile ./Makefile
ENV PATH=$PATH:/usr/local/go/bin
RUN make install_deps
COPY ./isoforest_microservice ./isoforest_microservice
# Build API Server
WORKDIR /app/isoforest_microservice/src
RUN 	GOOS=linux GOARCH=amd64 go build -trimpath -mod=readonly -gcflags="all=-spectre=all -N -l" -asmflags="all=-spectre=all" -ldflags="all=-s -w" -o /app/bin/isoforest_server.run main.go
# Run API Server
RUN useradd -ms /bin/bash go_user
USER go_user
CMD ["/bin/bash","-c", "/app/bin/isoforest_server.run"]
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 CMD [ "curl --location 'localhost:9001/status'" ]