#!/bin/bash

source /home/${USER}/.bashrc


lower_index=0
upper_index=10000
PREFIX=""
sim="squigulator"
window_size=(0)
VERSION=""
GPU="rtx2070"
DENSITY="5000"
DATA_HOME="$PAUL_NANOPORE_DATA/231220_experimental_fast5/dna/"
batch_names=$(ls -l  ${sequencing_main_path} | grep -Eo "23120(6|7).*dna_datastorage.*")                                                                   

while IFS= read -r line; do                                                                                                                                           
    batch_path=${DATA_HOME}/${line}/fast5_10k
    strand_names=$(ls -l  ${batch_path} | grep -Eo "strand[0-9]+_1[0]+\.fast5")
    while IFS = read -r strand; do
	echo "RUNNING window experiments"
	    for window in ${window_size[@]};
	    do
		real_strandId=$((strandId+1))
		fast5_path="strand_${real_strandId}_fast5"
		byte_index=$((strandId*16))
		echo ${byte_index}
		out_file=${PREFIX}${DENSITY}_"bonito_hedges_${sim}_strand${real_strandId}.fastq"
		err_file=${PREFIX}${DENSITY}_"bonito_hedges_${sim}_strand${real_strandId}.err"
		echo "--------------------------------------------------------------"
		echo "Strand ID ${real_strandId}"
		echo "Fast5 Path ${fast5_path}"
		echo "Index used ${byte_index}"
		echo "Output File ${out_file}"
		echo "Error File ${err_file}" 
		echo "--------------------------------------------------------------" 
		sbatch --time "24:00:00"  --exclude="c21,c32,c34,c[58-59],c68" --exclusive -J ${sim} -o ~/${out_file} -e ~/${err_file} -n 1  -p ${GPU}   --wrap "source ~/.bashrc && conda activate bonito_cuda && bonito basecaller dna_r9.4.1@v2 ${DATA_HOME}/${sim}/${DENSITY}_fast5/${fast5_path} --disable_koi --strand_pad GGCGACAGAAGAGTCAAGGTTC --hedges_bytes 204 ${byte_index} --hedges_params ~/hedges_options_${DENSITY}.json --disable_half --trellis base --mod_states 15 --processes 1 --lower_index ${lower_index} --upper_index ${upper_index} --batch_size 50 "
		#sbatch --time "24:00:00"  --exclude="c21,c32,c34,c[58-59],c68"  --exclusive -J ${sim} -o ~/basecaller_${out_file} -e ~/basecaller_${err_file} -n 1  -p ${GPU}   --wrap "source ~/.bashrc && conda activate bonito_cuda &&  bonito basecaller dna_r9.4.1@v2 ${DATA_HOME}/${sim}/${DENSITY}_fast5/${fast5_path} --disable_koi  --disable_half  --processes 1 --lower_index ${lower_index} --upper_index ${upper_index}"	
	    done
    done <<< "$strand_names"
done <<< "$batch_names"                                                                                                                                    
