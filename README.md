# NesT_HPML
Analysis of Nested Hierarchical Transformer on AI HW Kit with different optimization techniques

## GCP VM Instance Environment
**GPU:** `NVIDIA V100`  
**Image:** `Deep Learning VM for PyTorch 2.0 with CUDA 11.8 M114`  

## Setup

### AIHWKIT Instructions

The current method for installing **AIHWKIT** with *CUDA* support requires manual compilation, and there a lot of issues involved in the setup. Following the steps below worked for us

1. Install the required packages

    ```pip install --upgrade cmake``` 

    ```conda install openblas``` 

    ```pip install pybind11 scikit-build mypy timm wandb```  

    ```conda install -c intel mkl mkl-devel mkl-static mkl-include```

2. Clone the [official](https://github.com/IBM/aihwkit) aihwkit repository

    ```cd aihwkit```

    Edit lines `40-45` in the `CMakeLists.txt` as follows  
    ```# Append the virtualenv library path to cmake.  
    if(DEFINED ENV{VIRTUAL_ENV})
    include_directories("$ENV{VIRTUAL_ENV}/include")
    link_directories("$ENV{VIRTUAL_ENV}/lib")
    set(CMAKE_PREFIX_PATH "$ENV{VIRTUAL_ENV}")
    set(CMAKE_INCLUDE_PATH "$ENV{VIRTUAL_ENV}/include")
    endif()
    ```

3. Set the ENVIRONMENT variables

    ```export CMAKE_PREFIX_PATH="/opt/conda/pkgs/mkl-2024.0.0-intel_49656"```

4. Build

    ```make build_cuda flags="-DRPU_CUDA_ARCHITECTURES='70' -DINTEL_MKL_DIR='/opt/conda/pkgs/mkl-2024.0.0-intel_49656' -DCMAKE_INCLUDE_PATH='/opt/conda/pkgs/mkl-include-2024.0.0-intel_49656/include'"```

5. Set the PATH variable again before running the experiments (set for every new terminal)

    ```export LD_LIBRARY_PATH=/opt/conda/pkgs/mkl-2024.0.0-intel_49656/lib:$LD_LIBRARY_PATH```

## Commands

The simplest way (and convenient) to set configuration would be to update the `default.yaml` present in the `configs` directory. Each configuration would be automatically saved inside the `experiments/{run_name}` folder along with the **model weights**.

The command to run is  
`python main.py`

The metrics can be visualized at this [WandB](https://wandb.ai/hpmlugvc/NesT_HPML) space
