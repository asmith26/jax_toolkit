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

doc-generate-api:                       ## Generate API doc
	python docs/api_md_generate.py

doc-serve-locally: doc-generate-api     ## Serve docs locally
	firefox http://127.0.0.1:8000/
	mkdocs serve

lint:                                   ## Run lint checks
	tox -e lint

lint-fix:                               ## Automatically fix style violations
	black --line-length=120 jax_toolkit setup.py
	isort --lines 120 --recursive --use-parentheses jax_toolkit setup.py

pip-compile:                            ## Compile requirements.txt from setup.py
	pip-compile

pip-sync:                               ## Update your conda/virtual environment to reflect exactly (this will install/upgrade/uninstall everything necessary) what's in requirements.txt
	pip-sync

test-all: clean                         ## Run all checks with tox
	tox
