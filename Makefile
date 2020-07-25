help:                                   ## Show help docs
	@sed -ne '/@sed/!s/## //p' $(MAKEFILE_LIST)

bump2version-patch: test-all            ## Bump package patch version
	bump2version patch

bump2version-minor: test-all            ## Bump package minor version
	bump2version minor

bump2version-major: test-all            ## Bump package major version
	bump2version major

clean:                                  ## Remove all coverage, lint, test artifacts
	rm -f .coverage
	rm -rf .mypy_cache
	rm -rf .tox
	rm -rf *.egg-info

coverage:                               ## Create python test coverage report and open in firefox
	coverage run -m nose2 -v
	coverage report
	coverage html
	firefox htmlcov/index.html

doc-deploy: doc-generate-api            ## Deploy doc to github pages
	mkdocs gh-deploy

doc-serve-locally:				        ## Serve docs locally
	firefox http://127.0.0.1:8000/
	mkdocs serve

lint:                                   ## Run lint checks
	bandit --recursive jax_toolkit setup.py
	# PyCharm editor uses default 120 line length
	black --check --diff --line-length=120 jax_toolkit setup.py
	isort --check-only --lines 120 --multi-line=3 --recursive --trailing-comma --use-parentheses jax_toolkit setup.py
	flake8 --max-line-length=120 --extend-ignore=F401,I001 jax_toolkit setup.py
	mypy jax_toolkit --strict --ignore-missing-imports --implicit-optional --allow-untyped-decorators

lint-fix:                               ## Automatically fix style violations
	black --line-length=120 jax_toolkit setup.py
	isort --line-length=120 --profile=black jax_toolkit setup.py

pip-compile:                            ## Compile requirements.txt from setup.py
	pip-compile

pip-sync:                               ## Update your conda/virtual environment to reflect exactly (this will install/upgrade/uninstall everything necessary) what's in requirements.txt
	pip-sync

test-all: clean                         ## Run all checks with tox
	tox
