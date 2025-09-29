# Harpia

## Install

1. Build singularity
```
sudo -s singularity build harpia.sif container/Singularity.def
```

2. Access singularity. Replace '/path/to/harpia' with the directory where you downloaded the Harpia project.
```
singularity shell --nv -B /ibira /path/to/harpia/harpia.sif
bash
```

3. Create new environment
```
conda create -n harpia python=3.9 -y
```

4. Activate it
```
conda activate harpia
```

5. Install requirements
```
pip install -r requirements.txt
```

6. Install harpia
```
python3 setup.py build
pip install dist/harpia-2.3.3-cp39-cp39-linux_x86_64.whl
```

7. Check if installation was succeesfull
```
python3 tests_python/compilation_test.py
```

## Install-dev

1. Build singularity
```
sudo -s singularity build harpia.sif container/Singularity.def
```

2. Access singularity. Replace '/path/to/harpia' with the directory where you downloaded the Harpia project.
```
singularity shell --nv -B /ibira /path/to/harpia/harpia.sif
bash
```

3. Create new environment
```
conda create -n harpia-dev python=3.9 -y
```

4. Activate it
```
conda activate harpia-dev
```

5. Install requirements
```
pip install -r requirements-dev.txt
```

6. Install harpia
```
python3 setup.py build
pip install dist/harpia-2.3.3-cp39-cp39-linux_x86_64.whl
```

7. Check if installation was succeesfull
```
python3 tests_python/compilation_test.py
```

8. Install cucim dependencies:
   1. for cuda 11
   ```
   pip install cupy-cuda11x==13.5.1
   pip install --no-cache-dir --extra-index-url https://pypi.nvidia.com cucim-cu11==24.8.0
   ```
   2. for cuda 12
   ```
   pip install cupy-cuda12x==13.6.0
   pip install --no-cache-dir --extra-index-url https://pypi.nvidia.com cucim-cu12
   ```

