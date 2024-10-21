#!/usr/bin/env nextflow

process trainModel {
    container 'ghcr.io/eye2gene/e2g-train:latest'
    containerOptions " --gpus all "
    publishDir "${params.output_dir}", mode: 'copy'

    input:
    val model
    val epochs
    path train_csv
    path val_csv
    path cfg_63
    path baf_cfg
    path mini_cfg
    path "image_data" // mount images_data_dir as "image_data"
    val images_dir_in_csv
    val gpu

    output:
    path 'trained_models/*'

    """
    echo "Debug: Starting trainModel process"
    echo "Debug: Current working directory: \$(pwd)"
    echo "Debug: Listing current directory contents:"
    ls -la

    echo "Debug: Checking if running in Docker"
    if [ -f /.dockerenv ]; then
        echo "Debug: Running inside Docker"
    else
        echo "Debug: Not running inside Docker"
    fi

    echo "Debug: Checking CUDA and GPU"
    nvidia-smi 
    echo "Debug: CUDA version:"
    nvcc --version
    echo "Debug: Checking for libcuda.so:"
    ldconfig -p | grep libcuda
    echo "Debug: Checking LD_LIBRARY_PATH:"
    echo \$LD_LIBRARY_PATH

    echo "Debug: Python and TensorFlow versions:"
    python3 --version
    python3 -c "import tensorflow as tf; print(tf.__version__)"

    echo "Debug: TensorFlow GPU availability:"
    python3 -c "import tensorflow as tf; print(tf.test.is_built_with_cuda()); print(tf.test.is_gpu_available())"

    mkdir -p trained_models
    echo "Debug: Created trained_models directory"

    echo "Debug: Updating CSV files"
    # Update paths in CSV files
    sed -i 's|${images_dir_in_csv}|./image_data/|g' $train_csv
    sed -i 's|${images_dir_in_csv}|./image_data/|g' $val_csv

    echo "Debug: Starting training script"
    python3 /app/bin/train.py $model \
        --epochs $epochs \
        --train-dir $train_csv \
        --val-dir $val_csv \
        --cfg $cfg_63 $baf_cfg $mini_cfg \
        --gpu $gpu

    echo "Debug: Training script completed"
    """
}

workflow {
    trainModel(
        params.model, params.epochs,
        params.train_csv, params.val_csv,
        params.cfg_63, params.baf_cfg, params.mini_cfg,
        params.images_data_dir, params.images_dir_in_csv,
        params.gpu)
}
