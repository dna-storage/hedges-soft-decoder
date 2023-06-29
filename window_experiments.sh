#!/bin/bash



source /home/${USER}/.bashrc
module unload cuda
conda activate bonito_env


window_size=(100 300 500 800 1000 0)
strand_fast5=("221118_dna_volkel_strand1_fast5_10k_subset" "221118_dna_volkel_strand2_fast5_10k_subset")
strand_byte_index=(0 8)


echo "RUNNING window experiments"
for strandId in {0..1};
do
    for window in ${window_size[@]};
    do
	real_strandId=$((strandId+1))
	fast5_path=${strand_fast5[$strandId]}
	byte_index=${strand_byte_index[$strandId]}
	out_file="bonito_hedges_strand${real_strandId}_window--${window}.fastq"
	err_file="bonito_hedges_strand${real_strandId}_window--${window}.err"
	echo "--------------------------------------------------------------"
	echo "Strand ID ${real_strandId}"
	echo "Fast5 Path ${fast5_path}"
	echo "Index used ${byte_index}"
	echo "Output File ${out_file}"
	echo "Error File ${err_file}" 
	echo "--------------------------------------------------------------"
	sbatch --time "24:00:00" -J test -o $PAUL_NANOPORE_DATA/${out_file} -e $PAUL_NANOPORE_DATA/${err_file} -N 1 -n 1 -p rtx2070 --wrap "bonito basecaller dna_r9.4.1@v2 $PAUL_NANOPORE_DATA/${fast5_path} --disable_koi --strand_pad GGCGACAGAAGAGTCAAGGTTC --hedges_bytes 204 ${byte_index} --hedges_params $PAUL_NANOPORE_DATA/hedges_decode_ctc_debug/hedges_options.json --disable_half --window ${window} --trellis base"
    done
done


