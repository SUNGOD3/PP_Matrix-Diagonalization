# PP_Matrix-Diagonalization
This is a parallelized project aimed at accelerating the diagonalization of matrices through parallelization techniques. The project utilizes OpenMP to achieve a speedup of 2 to 4 times compared to the original code.

Refer to : https://github.com/SUNGOD3/PP_Matrix-Diagonalization/blob/main/.github/workflows/tests.yml
# Build ENV

```
git clone https://github.com/SUNGOD3/PP_Matrix-Diagonalization.git && cd PP_Matrix-Diagonalization
sudo apt-get -q update
sudo apt-get -qy install libomp-dev libtbb-dev
sudo apt-get install -y \
  curl build-essential make \
  gcc g++ intel-mkl-full \
  python3 python3-pip python3-pytest \
  python3-numpy python3-scipy python3-pandas

python3 -m pip install --upgrade pip
pip install pytest pytest-cov pybind11
```

# RUN

```
make test # for test
make demo # for demo
```
If you have some error in the build, you can git clone and push to your repo.
Then the github action will be invoked like https://github.com/SUNGOD3/PP_Matrix-Diagonalization/actions.
