help:                                        ## Show help docs
	@sed -ne '/@sed/!s/## //p' $(MAKEFILE_LIST)

bump2version-patch: test-all                 ## Bump package patch version
	bump2version patch

bump2version-minor: test-all                 ## Bump package minor version
	bump2version minor

bump2version-major: test-all                 ## Bump package major version
	bump2version major

clean:                                       ## Remove all coverage, lint, test artifacts
	rm -f .coverage
	rm -rf .mypy_cache
	rm -rf *.egg-info

coverage:                                    ## Run tests with coverage and report
	coverage run --omit="*/tests/*" -m nose2 -v
	coverage report

coverage-with-report:  coverage              ## Run tests with coverage and report, and open in firefox
	coverage html
	firefox htmlcov/index.html

doc-deploy: doc-generate-api                 ## Deploy doc to github pages
	mkdocs gh-deploy

doc-serve-locally:				             ## Serve docs locally
	firefox http://127.0.0.1:8000/
	mkdocs serve

lint:                                        ## Run lint checks
	bandit --recursive jax_toolkit setup.py
	# PyCharm editor uses default 120 line length
	black --check --diff --line-length=120 jax_toolkit setup.py
	isort --check-only --line-length=120 --profile=black jax_toolkit setup.py
	flake8 --max-line-length=120 --extend-ignore=F401,I001 jax_toolkit setup.py
	mypy jax_toolkit --strict --ignore-missing-imports --implicit-optional --allow-untyped-decorators

lint-fix:                                    ## Automatically fix style violations
	black --line-length=120 jax_toolkit setup.py
	isort --line-length=120 --profile=black jax_toolkit setup.py

pip-compile:                                 ## Compile requirements.txt from setup.py
	pip-compile

pip-install:                                 ## Install dependencies into current environment
	pip install -r requirements-dev.txt

pip-sync:                                    ## Update your conda/virtual environment to reflect exactly (this will install/upgrade/uninstall everything necessary) what's in requirements.txt
	pip-sync

pip-sync-dev: pip-sync pip-install           ## Update your conda/virtual environment to reflect exactly what's in requirements.txt, and then install the dev requirements.
	pip install .\[losses_utils\]

test-all: clean lint coverage-with-report    ## Run all checks
