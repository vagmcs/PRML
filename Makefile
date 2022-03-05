SHELL:=/usr/bin/env bash
.DEFAULT_GOAL := help

define LOGO

██████  ██████  ███    ███ ██
██   ██ ██   ██ ████  ████ ██
██████  ██████  ██ ████ ██ ██
██      ██   ██ ██  ██  ██ ██
██      ██   ██ ██      ██ ███████

endef

CURRENT_DIR := $(shell dirname $(realpath $(lastword $(MAKEFILE_LIST))))

$(info $(LOGO))

.PHONY: help
help:
	@echo "Usage: make command"
	@echo ""
	@echo "=== [Targets] ================================================="
	@sed -n 's/^###//p' < $(CURRENT_DIR)/Makefile | sort
	@echo "==============================================================="

### install: Install dependencies
.PHONY: install
install: clean
	@poetry install

### update:  Update dependencies
.PHONY: update
update: clean
	@poetry update

### pretty:  Format sources and apply code style
.PHONY: pretty
pretty:
	@poetry run isort .
	@poetry run black .

### compile: Apply code style and perform type checks
.PHONY: compile
compile: pretty
	@poetry check
	@poetry run flake8 --max-line-length 120 prml
	@poetry run mypy .

### jupyter: Start jupyter server
.PHONY: jupyter
jupyter:
	@poetry run jupyter notebook -y --log-level=INFO

### clean:   Clean the dependency cache and remove generated files
.PHONY: clean
clean:
	@poetry cache clear pypi --all -n
	@if [ -d "dist" ]; then rm -Rf $(CURRENT_DIR)/dist; fi
	@if [ -d ".generated" ]; then rm -Rf $(CURRENT_DIR)/.generated; fi