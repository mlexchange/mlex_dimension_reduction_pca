FROM python:3.11
COPY README.md README.md
COPY pyproject.toml pyproject.toml

RUN pip install --upgrade pip
RUN pip install .

WORKDIR /app/work/
COPY src/ src/
CMD ["bash"]
