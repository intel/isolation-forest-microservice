SHELL=/bin/bash

# Makefile for the ISOforest Microservice HTTP API Server
# Copyright 2023 Intel Corporation

# define standard colors
BLACK        := $(shell tput -Txterm setaf 0)
RED          := $(shell tput -Txterm setaf 1)
GREEN        := $(shell tput -Txterm setaf 2)
YELLOW       := $(shell tput -Txterm setaf 3)
LIGHTPURPLE  := $(shell tput -Txterm setaf 4)
PURPLE       := $(shell tput -Txterm setaf 5)
BLUE         := $(shell tput -Txterm setaf 6)
WHITE        := $(shell tput -Txterm setaf 7)

RESET := $(shell tput -Txterm sgr0)

os := $(shell uname -s)
arch := $(shell uname -p)
date := $(shell date)

echo:
	@printf "${GREEN}Make is installed. Build away!${RESET}\n"
ifeq (, $(shell which go))
	@printf "Golang not installed. Install go manually or use make install_deps.\n"
else
	@printf "${GREEN} Go installed! Ready to build.\n${RESET}"
endif

install_deps:
	@printf "${LIGHTPURPLE}Installing Go 1.24.6 64-bit...${RESET}\n"
	wget https://golang.org/dl/go1.24.6.linux-amd64.tar.gz
	tar -C /usr/local -xzf go1.24.6.linux-amd64.tar.gz
	@echo "export PATH=$$PATH:/usr/local/go/bin" >> ~/.bashrc
	exec $$SHELL

.PHONY: build_linux
build_linux: start_build clean
ifeq ($(arch),x86_64)
	@printf "\t${PURPLE}Building for Linux 64-bit...${RESET}\n"
	cd isoforest_microservice/src; \
	GOOS=linux GOARCH=amd64 go build -trimpath -mod=readonly -gcflags="all=-spectre=all -N -l" -asmflags="all=-spectre=all" -ldflags="all=-s -w" -o ../../bin/isoforest_server.run main.go

	@printf "\t${GREEN}Linux 64 build completed.${RESET}\n"
endif

.PHONY: build_linux_container
build_linux_container: start_build clean
ifeq ($(arch),x86_64)
	@printf "\t${PURPLE}Building for Linux 64-bit...${RESET}\n"
	GOOS=linux GOARCH=amd64 go build -trimpath -mod=readonly -gcflags="all=-spectre=all -N -l" -asmflags="all=-spectre=all" -ldflags="all=-s -w" -o /app/bin/isoforest_server.run main.go

	@printf "\t${GREEN}Linux 64 build completed.${RESET}\n"
endif

.PHONY: start_build
start_build:
	@printf "\t${GREEN}Starting Build${RESET} Current Time: $(shell date +"%T.%3N")\n"
	@printf "\t${BLUE}Executing tests...${RESET}\n"
	-go test ../...


.PHONY: compile_assets
compile_assets:
	@printf "\t${BLUE}Compiling assets...${RESET}\n"
	go run ../internal/vfsgen/asset_gen.go
	mv apiassets.go ../pkg/apiserver/apiassets.go
	mv mainassets.go ../pkg/main/mainassets.go
	mv pamassets.go ../pkg/ncbpam/pamassets.go

.PHONY:full_build
full_build: start_build run_tests build_linux finish_build

.PHONY:run_tests
run_tests: 
	@printf "\t${YELLOW}Running Test Suite...${RESET}\n"
	go test -cover -covermode=atomic -coverprofile=main_coverage.out -cpuprofile=main_cpu.out -memprofile=main_mem.out -coverpkg=all -outputdir=../tests/ -v -o ../tests/main.test ../pkg/main
	go test -cover -covermode=atomic -coverprofile=apiserver_coverage.out -cpuprofile=apiserver_cpu.out -memprofile=apiserver_mem.out -coverpkg=all -outputdir=../tests/ -v -o ../tests/apiserver.test ../pkg/apiserver
	go test -cover -covermode=atomic -coverprofile=projectpath_coverage.out -cpuprofile=projectpath_cpu.out -memprofile=projectpath_mem.out -coverpkg=all -outputdir=../tests/ -v -o ../tests/projectpath.test ../pkg/projectpath
	go test -cover -covermode=atomic -coverprofile=logrusfilehook_coverage.out -cpuprofile=logrusfilehook_cpu.out -memprofile=logrusfilehook_mem.out -coverpkg=all -outputdir=../tests/ -v -o ../tests/logrusfilehook.test ../pkg/logrusfilehook
	go test -cover -covermode=atomic -coverprofile=ncbpam_coverage.out -cpuprofile=ncbpam_cpu.out -memprofile=ncbpam_mem.out -coverpkg=all -outputdir=../tests/ -v -o ../tests/ncbpam.test ../pkg/ncbpam
	go tool cover -html=../tests/main_coverage.out -o=../tests/main_coverage.html
	go tool cover -html=../tests/apiserver_coverage.out -o=../tests/apiserver_coverage.html
	go tool cover -html=../tests/projectpath_coverage.out -o=../tests/projectpath_coverage.html
	go tool cover -html=../tests/logrusfilehook_coverage.out -o=../tests/logrusfilehook_coverage.html
	go tool cover -html=../tests/ncbpam_coverage.out -o=../tests/ncbpam_coverage.html

.PHONY:run
run:
	./bin/isoforest_server.run


.PHONY:run_debug
run_debug:
	./bin/isoforest_server.run -debug

.PHONY:run_noencrypt
run_noencrypt:
	./bin/isoforest_server.run -noencrypt

.PHONY:run_nolog
run_nolog:
	./bin/isoforest_server.run -nolog

.PHONY:run_options
run_options:
	./bin/isoforest_server.run -h

.PHONY:run_background
run_background:
	./bin/isoforest_server.run > /dev/null 2>&1 &

.PHONY:clean
clean:
ifeq ("$(wildcard $(./bin/isoforest_server.run))","")
	-rm ./bin/isoforest_server.run
endif
	@printf "\t${RED}Binaries removed${RESET}\n"

.PHONY:help
help:
	@printf "\t${GREEN}Build and run the server.${RESET} Usage: make ${YELLOW}<Target>${RESET}\n Targets:\n"
	@printf "\t${BLUE}compile_assets${RESET}: compiles assets and configuration files for bundling with the binary. Necessary for single-binary distribution.\n"
	@printf "\t${BLUE}full_build${RESET}: Runs a full build. Builds for ARM64, Linux x86_64, runs the testing suite, generates profiles and test binaries.\n"
	@printf "\t${BLUE}install_deps${RESET}: attempts to install Go 1.23.3.\n"
	@printf "\t${BLUE}echo${RESET}: checks for make and go\n"
	@printf "\t${BLUE}run_tests${RESET}: runs the Go test suite, creating CPU, Memory, and Test profiles and a Test Coverage HTML report. Note that this is computationally expensive.\n"
	@printf "\t${BLUE}run${RESET}: run the server with default settings. Serves the server on ${GREEN}localhost:8080${RESET}. If using an external device to access, use the server's ip address instead of localhost.\n"
	@printf "\t${BLUE}run_debug${RESET}: run the server in debug mode. ${RED}Extremely insecure! Only use for dev purposes, and even then use only when absolutely necessary!${RESET} ${YELLOW}Username: admin Password: admin ${RESET}\n"
	@printf "\t${BLUE}run_nolog${RESET}: run the server without logging to file. ${YELLOW}Not recommended. ${RESET}\n"
	@printf "\t${BLUE}run_options${RESET}: invoke's the server help function, explaining the possible command flags.\n"
	@printf "\t${BLUE}run_background${RESET}: run the server in the background. The process ID will be printed on invoking.\n"
	@printf "\t${BLUE}clean${RESET}: clear out the binaries from the ../bin/ directory.\n"
