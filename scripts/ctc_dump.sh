#!/bin/bash
source /home/${USER}/.bashrc


lower_index=0
upper_index=10000
PREFIX=""
sim="DeepSimulator"
window_size=(0)
VERSION=""
GPU="rtx2060super"
DENSITY="1667"
DATA_HOME="/mnt/beegfs/kvolkel/221118_nanopore_dna"
strands=($(ls ${DATA_HOME}/${sim}/${DENSITY}_fast5/| grep -Eo "_[1-2]_" | grep -Eo "[1-2]"))

echo "RUNNING ctc experiments"
for s in ${strands[@]};
do
    strandId=$((s-1))
    for window in ${window_size[@]};
    do
	real_strandId=$((strandId+1))
	fast5_path="strand_${real_strandId}_fast5"
	byte_index=$((strandId*8))
	echo ${byte_index}
	out_file=${PREFIX}${DENSITY}_"bonito_ctc_${sim}_strand${real_strandId}.fastq"
	err_file=${PREFIX}${DENSITY}_"bonito_ctc_${sim}_strand${real_strandId}.err"
	h5_file=${PREFIX}${DENSITY}_"bonito_ctc_${sim}_strand${real_strandId}.hdf5"
	echo "--------------------------------------------------------------"
	echo "Strand ID ${real_strandId}"
	echo "Fast5 Path ${fast5_path}"
	echo "Index used ${byte_index}"
	echo "Output File ${out_file}"
	echo "Error File ${err_file}" 
	echo "--------------------------------------------------------------" 
	sbatch --time "24:00:00"  --exclude="c21,c32,c34,c[58-59],c68" --exclusive -J ${sim} -o ~/${out_file} -e ~/${err_file} -n 1  -p ${GPU}   --wrap "source ~/.bashrc && conda activate bonito_cuda && bonito basecaller dna_r9.4.1@v2 ${DATA_HOME}/${sim}/${DENSITY}_fast5/${fast5_path} --disable_koi --strand_pad GGCGACAGAAGAGTCAAGGTTC --hedges_bytes 204 ${byte_index} --hedges_params ~/hedges_options_${DENSITY}.json --disable_half --trellis base  --processes 1 --lower_index ${lower_index} --upper_index ${upper_index} --batch_size 1  --ctc_fast5 /home/kvolkel/${h5_file}"
    done
done


