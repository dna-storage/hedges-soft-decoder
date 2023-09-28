#!/bin/bash

source /home/${USER}/.bashrc
module unload cuda
conda activate bonito_cuda

PREFIX=""
batch_size=(2 4 8 16 32 64 100)

GPU="rtx2060super"
exclude="--exclude=\"c21,c32,c34,c[58-59],c68\""



echo "RUNNING batch benchmark experiments"

    for batch in ${batch_size[@]};
    do
	out_file=${PREFIX}"bonito_hedges_benchmark_batch--${batch}.fastq"
	err_file=${PREFIX}"bonito_hedges_benchmark_batch--${batch}.err"
	echo "--------------------------------------------------------------"
	echo "Strand ID ${real_strandId}"
	echo "Fast5 Path ${fast5_path}"
	echo "Index used ${byte_index}"
	echo "Output File ${out_file}"
	echo "Error File ${err_file}" 
	echo "--------------------------------------------------------------"
	sbatch --time "24:00:00" ${exclude}  -J benchmark_batch -o $PAUL_NANOPORE_DATA/${out_file} -e $PAUL_NANOPORE_DATA/${err_file} -N 1 -p ${GPU} --wrap "python debug/hedges_ctc_debug.py $PAUL_NANOPORE_DATA/strand_1_benchmark_scores $PAUL_NANOPORE_DATA/hedges_decode_ctc_debug/hedges_options.json --batch ${batch} --trellis base"
    done
