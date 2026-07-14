.PHONY: validate validate-docker clean-junk

validate: validate-docker

validate-docker:
	./scripts/validate_ci_local.sh

clean-junk:
	find . -name .DS_Store -type f -delete
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
	rm -rf .pytest_cache .ruff_cache .mypy_cache
