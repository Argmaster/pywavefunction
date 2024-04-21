# Build from source

To build Pywavefunction from source You have to set up [Development](#development)
environment first. Make sure you have `poetry` environment activated with:

```
poetry shell
```

With environment active it should be possible to build wheel and source distribution
with:

```
poetry build
```

Check `dist` directory within current working directory, `pywavefunction-x.y.z.tar.gz`
and `pywavefunction-x.y.z-py3-none-any.whl` should be there.
