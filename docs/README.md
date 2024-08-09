# README

## Requirements

Install dependencies

```
pip install .[dev]
```

Use `sphinx-apidoc` to help automatically generate sources and document a whole python project.
This should only need to be done once.

```
sphinx-apidoc -o source -f .. ../client_test.py ../predictionguard/version.py
```

The `-f ...` excludes a few dirs/files from the api docs that we don't want to include.

## Build

```
make html
```

Serve and preview docs locally

```
cd build/html
python3 -m http.server
```

Open your browser to localhost:8000.

## Publish to Github Pages

On tagged releases, Github Actions will build and publish docs to Github pages by uploading an artifact.
The workflow is defined in [.github/workflows/docs.yml](../.github/workflows/docs.yml).

