# README

Install dependencies

```
pip install sphinx
pip install sphinx-autodoc-typehints
```

Use `sphinx-apidoc` to help automatically generate sources and document a whole python project.

```
sphinx-apidoc -o source -f .. ../client_test.py ../predictionguard/version.py
```

Serve docs locally

```
cd build/html
python3 -m http.server
```
