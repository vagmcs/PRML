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

### notes:   Create PDF from notebooks
.PHONY: notes
notes:
	@cd notebooks; \
  	nbmerge \
  	ch1_introduction.ipynb \
  	ch2_probability_distributions.ipynb \
  	ch3_linear_models_for_regression.ipynb \
	ch4_linear_models_for_classification.ipynb \
	ch5_neural_networks.ipynb > PRML.ipynb; \
	jupyter-nbconvert \
	--log-level CRITICAL \
	--to latex PRML.ipynb; \
	sed 's/section/section*/' \
	prml.tex > prml_no_sections.tex; \
	xelatex prml_no_sections.tex >/dev/null; \
	rm -r prml.ipynb *.aux *.out *.log *.tex PRML_files >/dev/null; \
	mv prml_no_sections.pdf ../PRML.pdf

### clean:   Clean the dependency cache and remove generated files
.PHONY: clean
clean:
	@poetry cache clear pypi --all -n
	@if [ -d "dist" ]; then rm -Rf $(CURRENT_DIR)/dist; fi