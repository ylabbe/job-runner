module purge

source $CONDA_ROOT/bin/activate
conda activate $CONDA_ENV
cd $PROJECT_DIR

echo CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES
echo EGL_VISIBLE_DEVICES: $EGL_VISIBLE_DEVICES
