#!/usr/bin/env nextflow

process trainModel {
    container 'eye2gene/e2g-train'
    containerOptions "--gpus all"
    accelerator 1
    memory '16 GB'

    publishDir "${params.output_dir}", mode: 'copy'

    input:
    val model
    val epochs
    path train_csv
    path val_csv
    path load_weights_h5_path
    path train_config
    path "image_data" // mount images_data_dir as "image_data"
    val images_dir_in_csv
    val gpu

    output:
    path 'trained_models/*'
    path "checkpoints/*"
    path "logs/*"

    script:
    def tmp = []
    if (load_weights_h5_path) tmp.add('--resume-from')
    if (load_weights_h5_path) tmp.add(load_weights_h5_path[0])
    def extra_args = tmp.join(' ')
    """
    echo "Debug: Updating CSV files"
    # Update paths in CSV files
    sed -i 's|${images_dir_in_csv}|./image_data/|g' $train_csv
    sed -i 's|${images_dir_in_csv}|./image_data/|g' $val_csv

    mkdir -p trained_models

    echo "Debug: Starting training script"
    python3 /app/bin/train.py $model --model-save-dir trained_models \
        --epochs $epochs --train-dir $train_csv --val-dir $val_csv \
        --cfg $train_config --gpu $gpu $extra_args

    echo "Debug: Training script completed"
    """
}

workflow {
    load_paths = params.load_weights_h5_path != '' ? [params.load_weights_h5_path] : []
    trainModel(
        params.model, params.epochs,
        params.train_csv, params.val_csv, load_paths,
        params.train_config,
        params.images_data_dir, params.images_dir_in_csv,
        params.gpu)
}
