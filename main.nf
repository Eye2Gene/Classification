#!/usr/bin/env nextflow

process trainModel {
    container 'e2g-train'
    containerOptions '--gpus all'
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
    mkdir -p trained_models

    # Update paths in CSV files
    sed -i 's|${images_dir_in_csv}|./image_data/|g' $train_csv
    sed -i 's|${images_dir_in_csv}|./image_data/|g' $val_csv

    python3 /app/bin/train.py $model \
        --epochs $epochs \
        --train-dir $train_csv \
        --val-dir $val_csv \
        --cfg $cfg_63 $baf_cfg $mini_cfg \
        --gpu $gpu
    """
}

workflow {
    trainModel(
        params.model, params.epochs,
        params.train_csv, params.val_csv,
        file(params.cfg_63), file(params.baf_cfg), file(params.mini_cfg),
        params.images_data_dir, params.images_dir_in_csv
        params.gpu)
}