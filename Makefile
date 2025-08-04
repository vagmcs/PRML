SHELL:=/usr/bin/env bash
.DEFAULT_GOAL := help

define LOGO

██████  ██████  ███    ███ ██
██   ██ ██   ██ ████  ████ ██
██████  ██████  ██ ████ ██ ██
██      ██   ██ ██  ██  ██ ██
██      ██   ██ ██      ██ ███████

endef

CURRENT_DIR:=$(shell dirname $(realpath $(lastword $(MAKEFILE_LIST))))
PROJECT_NAME:=$(shell poetry version | sed -e 's/[ ].*//g' | tr '-' '_')
PROJECT_VERSION:=$(shell poetry version | sed -e 's/.*[ ]//g')

$(info $(LOGO))

.PHONY: help
help:
	@echo "Usage: make command"
	@echo ""
	@echo "=== [Targets] ================================================="
	@sed -n 's/^###//p' < $(CURRENT_DIR)/Makefile | sort
	@echo "==============================================================="

### clean    : Clean the dependency cache and remove generated files
.PHONY: clean
clean:
	@poetry cache clear pypi --all -n
	@if [ -d "dist" ]; then rm -Rf $(CURRENT_DIR)/dist; fi

### format   : Format source
.PHONY: format
format:
	@poetry run pyupgrade --py39-plus **/*.py || true
	@poetry run nbqa pyupgrade --py39-plus **/*.ipynb || true
	@poetry run isort $(PROJECT_NAME)
	@poetry run nbqa isort notebooks --float-to-top
	@poetry run ruff format notebooks $(PROJECT_NAME)
	@poetry run docformatter $(PROJECT_NAME)|| true

### compile  : Apply code styling and perform type checks
.PHONY: lint
lint:
	@poetry check
	@poetry run ruff check --fix notebooks $(PROJECT_NAME)
	@poetry run mypy $(PROJECT_NAME)

### jupyter  : Start jupyter server
.PHONY: jupyter
jupyter:
	@poetry run jupyter notebook -y --log-level=INFO

### notes    : Create PDF from notebooks
.PHONY: notes
notes:
	@cd notebooks; \
  	nbmerge \
  	ch1_introduction.ipynb \
  	ch2_probability_distributions.ipynb \
  	ch3_linear_models_for_regression.ipynb \
	ch4_linear_models_for_classification.ipynb \
	ch5_neural_networks.ipynb \
	gradient_descent_algorithms.ipynb \
	ch6_kernel_methods.ipynb \
	ch7_sparse_kernel_machines.ipynb \
	ch9_mixture_models_and_em.ipynb > PRML.ipynb; \
	jupyter-nbconvert \
	--log-level CRITICAL \
	--to latex PRML.ipynb; \
	sed 's/section/section*/' \
	PRML.tex > prml_no_sections.tex; \
	xelatex prml_no_sections.tex >/dev/null; \
	rm -r prml.ipynb *.aux *.out *.log *.tex PRML_files >/dev/null; \
	mv prml_no_sections.pdf ../PRML.pdf

### markdown : Create Markdown from notebooks
.PHONY: markdown
markdown:
	@cd notebooks; \
	jupyter-nbconvert \
	--log-level CRITICAL \
	--output-dir=md \
	--to markdown ch1_introduction.ipynb \
	ch2_probability_distributions.ipynb \
	ch3_linear_models_for_regression.ipynb \
	ch4_linear_models_for_classification.ipynb \
	ch5_neural_networks.ipynb \
	gradient_descent_algorithms.ipynb \
	ch6_kernel_methods.ipynb \
	ch7_sparse_kernel_machines.ipynb \
	ch9_mixture_models_and_em.ipynb;