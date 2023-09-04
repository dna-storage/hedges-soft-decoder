#!/bin/bash

source /home/${USER}/.bashrc
module unload cuda
conda activate bonito_cuda


lower_index=0
upper_index=10000
PREFIX=""
window_size=(0)
VERSION=""
GPU="rtx2060super"


echo "RUNNING window experiments"
for strandId in {0..3};
do
    for window in ${window_size[@]};
    do
	real_strandId=$((strandId+1))
	fast5_path=strand_${real_strand_ID}_fast5
	byte_index=$((real_strandID*8))
	out_file=${PREFIX}"bonito_hedges_squigulator_strand${real_strandId}.fastq"
	err_file=${PREFIX}"bonito_hedges_squigulator_strand${real_strandId}.err"
	echo "--------------------------------------------------------------"
	echo "Strand ID ${real_strandId}"
	echo "Fast5 Path ${fast5_path}"
	echo "Index used ${byte_index}"
	echo "Output File ${out_file}"
	echo "Error File ${err_file}" 
	echo "--------------------------------------------------------------"
	sbatch --time "24:00:00" --exclude="c[78-94],c[60]" -J squigulator -o ~/${out_file} -e ~/${err_file} -N 1 -p ${GPU} --wrap "bonito basecaller dna_r9.4.1@v2 ~/squigulator/${fast5_path} --disable_koi --strand_pad GGCGACAGAAGAGTCAAGGTTC --hedges_bytes 204 ${byte_index} --hedges_params ~/hedges_options.json --disable_half --trellis base --processes 1 --lower_index ${lower_index} --upper_index ${upper_index} --batch 50"
    done
done


