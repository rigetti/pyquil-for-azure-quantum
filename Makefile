dev:
	make style
	make lint
	make test-all

style:
	isort .
	black .

check-style:
	isort --check .
	black --check .

lint:
	pylint pyquil_azure_quantum
	pylint test
	mypy .

test-all:
	pytest

test-no-qcs:
	pytest -k "not test_e2e_qcs_operations"

test-requires-qcs:
	pytest -k test_e2e_qcs_operations