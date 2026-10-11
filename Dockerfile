FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
WORKDIR /app

COPY pyproject.toml README.md LICENSE ./
COPY aichain ./aichain
COPY personas ./personas
# Editable install keeps personas/ next to the package, where prompt.py looks for it.
RUN pip install -e .

RUN useradd --create-home --uid 10001 aichain && mkdir -p /app/data && chown aichain /app/data
USER aichain

ENTRYPOINT ["python", "-m", "aichain.sim.demo"]
CMD ["--quiet"]
