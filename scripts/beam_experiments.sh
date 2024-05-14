#!/bin/bash                                                                                                                                                           

source /home/${USER}/.bashrc
module unload cuda
conda activate bonito_cuda

PREFIX=""
window_size=0
fast5_top="221118_dna_nanopore_experimental_2k"
density="1667"
strand_fast5=("strand_1_fast5" "strand_2_fast5")
declare -a li_strand1=($(seq 0 200 1800))
declare -a li_strand2=($(seq 0 200 1800))
declare -a lower_indexes=("(${li_strand1[*]@Q})" "(${li_strand2[*]@Q})")

declare -a ui_strand1=($(seq 200 200 2000))
declare -a ui_strand2=($(seq 200 200 2000))
declare -a upper_indexes=("(${ui_strand1[*]@Q})" "(${ui_strand2[*]@Q})")

strand_byte_index=(0 8)
GPU=rtx2060super

exclude="--exclude=c21,c32,c34,c[58-59],c68"

echo "RUNNING beam experiments"
for strandId in {0..1};
do
    declare -a lower_index_list=${lower_indexes[$strandId]}
    declare -a upper_index_list=${upper_indexes[$strandId]}
    for (( indexId=0; indexId<${#lower_index_list[@]}; indexId++));
    do
            real_strandId=$((strandId+1))
            fast5_path=${fast5_top}/${density}"_fast5"/${strand_fast5[$strandId]}
            byte_index=${strand_byte_index[$strandId]}
	    lower_index=${lower_index_list[$indexId]}
	    upper_index=${upper_index_list[$indexId]}
            out_file=${PREFIX}"bonito_hedges_strand${real_strandId}_beam_${lower_index}-${upper_index}.fastq"
            err_file=${PREFIX}"bonito_hedges_strand${real_strandId}_beam_${lower_index}-${upper_index}.err"
            echo "--------------------------------------------------------------"
            echo "Strand ID ${real_strandId}"
            echo "Fast5 Path ${fast5_path}"
            echo "Index used ${byte_index}"
            echo "Output File ${out_file}"
            echo "Error File ${err_file}"
	    echo "Lower Index $lower_index"
	    echo "Upper Index $upper_index"
            echo "--------------------------------------------------------------"
            sbatch --time "24:00:00" ${exclude}  -J beam_experiments -o $PAUL_NANOPORE_DATA/${out_file} -e $PAUL_NANOPORE_DATA/${err_file} -N 1 -n 1 -p ${GPU} --wrap "bonito basecaller dna_r9.4.1@v2 ${PAUL_NANOPORE_DATA}/${fast5_path} --disable_koi --strand_pad GGCGACAGAAGAGTCAAGGTTC --hedges_bytes 204 ${byte_index} --hedges_params $PAUL_NANOPORE_DATA/hedges_decode_ctc_debug/hedges_options.json --disable_half --window ${window_size} --trellis beam_1 --lower_index ${lower_index} --upper_index ${upper_index} --processes 1"
    done
done
