#!/bin/bash

source /home/${USER}/.bashrc
module unload cuda
conda activate bonito_cuda

PREFIX=""
window_size=(0)

GPU="rtx2060super"
exclude="--exclude=c21,c32,c34,c[58-59],c68"

echo "RUNNING window benchmark experiments"

    for window in ${window_size[@]};
    do
	out_file=${PREFIX}"bonito_hedges_benchmark_window--${window}.fastq"
	err_file=${PREFIX}"bonito_hedges_benchmark_window--${window}.err"
	echo "--------------------------------------------------------------"
	echo "Strand ID ${real_strandId}"
	echo "Fast5 Path ${fast5_path}"
	echo "Index used ${byte_index}"
	echo "Output File ${out_file}"
	echo "Error File ${err_file}" 
	echo "--------------------------------------------------------------"
	echo sbatch --time "24:00:00" --exclude="c[78-94],c[60]" -J benchmark -o $PAUL_NANOPORE_DATA/${out_file} -e $PAUL_NANOPORE_DATA/${err_file} -N 1 -p ${GPU} --wrap "python debug/hedges_ctc_debug.py $PAUL_NANOPORE_DATA/strand_1_benchmark_scores $PAUL_NANOPORE_DATA/hedges_decode_ctc_debug/hedges_options.json --window_size ${window} --trellis base --mod_states 7"
    done


mods=(3 7 15)
echo "RUNNING mod benchmark experiments"
    for mod in ${mods[@]};
    do
	out_file=${PREFIX}"bonito_hedges_benchmark_mod--${mod}.fastq"
	err_file=${PREFIX}"bonito_hedges_benchmark_mod--${mod}.err"
	echo "--------------------------------------------------------------"
	echo "Strand ID ${real_strandId}"
	echo "Fast5 Path ${fast5_path}"
	echo "Index used ${byte_index}"
	echo "Output File ${out_file}"
	echo "Error File ${err_file}" 
	echo "--------------------------------------------------------------"
	echo sbatch --time "24:00:00" --exclude="c[78-94],c[60]" -J benchmark -o $PAUL_NANOPORE_DATA/${out_file} -e $PAUL_NANOPORE_DATA/${err_file} -N 1 -p ${GPU} --wrap "python debug/hedges_ctc_debug.py $PAUL_NANOPORE_DATA/strand_1_benchmark_scores $PAUL_NANOPORE_DATA/hedges_decode_ctc_debug/hedges_options.json --window_size 0 --trellis mod --mod_states ${mod}"
    done



echo "RUNNING beam benchmark experiments"
	out_file=${PREFIX}"bonito_hedges_benchmark_beam.fastq"
	err_file=${PREFIX}"bonito_hedges_benchmark_beam.err"
	echo "--------------------------------------------------------------"
	echo "Strand ID ${real_strandId}"
	echo "Fast5 Path ${fast5_path}"
	echo "Index used ${byte_index}"
	echo "Output File ${out_file}"
	echo "Error File ${err_file}" 
	echo "--------------------------------------------------------------"
	sbatch --time "24:00:00" ${exclude} -J benchmark -o $PAUL_NANOPORE_DATA/${out_file} -e $PAUL_NANOPORE_DATA/${err_file} -N 1 -p ${GPU} --wrap "python debug/hedges_ctc_debug.py $PAUL_NANOPORE_DATA/strand_1_benchmark_scores $PAUL_NANOPORE_DATA/hedges_decode_ctc_debug/hedges_options.json --trellis beam_1 "
 

