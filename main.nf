#!/usr/bin/env nextflow

process trainModel {
    container 'ghcr.io/eye2gene/e2g-train:latest'
    containerOptions "--gpus all"
    accelerator 1
    memory '16 GB'

    publishDir "${params.output_dir}", mode: 'copy'

    input:
    val model
    val epochs
    path train_csv
    path val_csv
    path model_save_dir
    path cfg_63
    path baf_cfg
    path mini_cfg
    path "image_data" // mount images_data_dir as "image_data"
    val images_dir_in_csv
    val gpu

    output:
    path 'trained_models/*'

    """
    echo "Debug: Updating CSV files"
    # Update paths in CSV files
    sed -i 's|${images_dir_in_csv}|./image_data/|g' $train_csv
    sed -i 's|${images_dir_in_csv}|./image_data/|g' $val_csv

    echo "Debug: Starting training script"
    python3 /app/bin/train.py $model \
        --model-save-dir $model_save_dir \
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
        params.train_csv, params.val_csv, params.model_save_dir,
        params.cfg_63, params.baf_cfg, params.mini_cfg,
        params.images_data_dir, params.images_dir_in_csv,
        params.gpu)
}
