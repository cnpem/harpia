# Harpia

## Install

1. Create new environment
```
conda create -n harpia python=3.9 -y
```

2. Activate it
```
conda activate harpia
```

3. Install requirements
```
pip install -r requirements.txt
```

4. Install harpia
```
python3 setup.py build
pip install dist/harpia-2.3.3-cp39-cp39-linux_x86_64.whl
```

5. Check if installation was succeesfull
```
python3 tests_python/compilation_test.py
```

## Install-dev

1. Create new environment
```
conda create -n harpia-dev python=3.9 -y
```

2. Activate it
```
conda activate harpia-dev
```

3. Install requirements
```
pip install -r requirements-dev.txt
```

4. Install harpia
```
python3 setup.py build
pip install dist/harpia-2.3.3-cp39-cp39-linux_x86_64.whl
```

5. Check if installation was succeesfull
```
python3 tests_python/compilation_test.py
```

6. Install cucim dependencies:
   1. for cuda 11
   ```
   pip install cupy-cuda11x==13.5.1
   pip install --no-cache-dir --extra-index-url https://pypi.nvidia.com cucim-cu11==24.8.0
   ```
   2. for cuda 12
   ```
   pip install cupy-cuda12x==12.4.0 cucim-cu12==23.10.0
   ```

