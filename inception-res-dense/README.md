# Inception / Res / Dense

This project explores three different model architecture wit the `Cifar10` dataset.

## Environment
Here is the structure of the .`env` file.
```env
ARTIFACTS_DIR=/path/to/artifacts-dir

# optional
DB_BASE_DIR=/path/to/base-dir
```

## Using the Torch Backend
If the the torch backend does not work, and lib-torch was installed manually, ensure that the `LIBTORCH` and `LD_LIBRARY_PATH` environment variables are set.
```bash
$ export LIBTORCH=/path/to/libtorch-2.1.0/
$ export LD_LIBRARY_PATH=/path/to/libtorch/lib:$LD_LIBRARY_PATH
```


