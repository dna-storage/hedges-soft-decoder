#!/bin/bash                                                                                                                                                           

source /home/${USER}/.bashrc
module unload cuda
conda activate bonito_env


window_size=0
mods=(15)
strand_fast5=("221118_dna_volkel_strand1_fast5_10k_subset" "221118_dna_volkel_strand2_fast5_10k_subset")
declare -a li_strand1=(0 2000 4000 6000 8000)
declare -a li_strand2=(0 2000 4000 6000 8000)
declare -a lower_indexes=("(${li_strand1[*]@Q})" "(${li_strand2[*]@Q})")

declare -a ui_strand1=(2000 4000 6000 8000 10000)
declare -a ui_strand2=(2000 4000 6000 8000 10000)
declare -a upper_indexes=("(${ui_strand1[*]@Q})" "(${ui_strand2[*]@Q})")

strand_byte_index=(0 8)
GPU=rtx2060super

echo "RUNNING mod experiments"
for strandId in {0..1};
do
    declare -a lower_index_list=${lower_indexes[$strandId]}
    declare -a upper_index_list=${upper_indexes[$strandId]}
    for (( indexId=0; indexId<${#lower_index_list[@]}; indexId++));
    do
	for mod in ${mods[@]};
	do
            real_strandId=$((strandId+1))
            fast5_path=${strand_fast5[$strandId]}
            byte_index=${strand_byte_index[$strandId]}
	    lower_index=${lower_index_list[$indexId]}
	    upper_index=${upper_index_list[$indexId]}
            out_file="bonito_hedges_strand${real_strandId}_window--${window_size}_mod--${mod}_${lower_index}-${upper_index}.fastq"
            err_file="bonito_hedges_strand${real_strandId}_window--${window_size}_mod--${mod}_${lower_index}-${upper_index}.err"
            echo "--------------------------------------------------------------"
            echo "Strand ID ${real_strandId}"
            echo "Fast5 Path ${fast5_path}"
            echo "Index used ${byte_index}"
            echo "Output File ${out_file}"
            echo "Error File ${err_file}"
	    echo "Lower Index $lower_index"
	    echo "Upper Index $upper_index"
            echo "--------------------------------------------------------------"
            sbatch   --time "24:00:00" -J mod_experiments_${window_size}_${mod} -o $PAUL_NANOPORE_DATA/${out_file} -e $PAUL_NANOPORE_DATA/${err_file} -N 1 -n 1 -p ${GPU} --wrap "bonito basecaller dna_r9.4.1@v2 $PAUL_NANOPORE_DATA/${fast5_path} --disable_koi --strand_pad GGCGACAGAAGAGTCAAGGTTC --hedges_bytes 204 ${byte_index} --hedges_params $PAUL_NANOPORE_DATA/hedges_decode_ctc_debug/hedges_options.json --disable_half --window ${window_size} --trellis mod --mod_states ${mod} --lower_index ${lower_index} --upper_index ${upper_index}"
	done
    done
done
