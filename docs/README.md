# README

Use `sphinx-apidoc` to help automatically generate sources and document a whole python project.

```
sphinx-apidoc -o source -f .. ../client_test.py
```

Serve docs locally

```
cd build/html
python3 -m http.serer
```
