#!/bin/bash
source /home/${USER}/.bashrc


lower_index=0
upper_index=10000
PREFIX=""
#sim="221118_dna_nanopore_experimental"
#sim="squigulator"
sim="DeepSimulator"
window_size=(0)
VERSION=""
GPU="rtx2070"
DENSITY="1667"
H5_DIR="/home/kvolkel/231004_score_hdf5"
DNA_HOME="/home/kvolkel/squigulator"
#1-base indexing
strands=(1 2)

echo "RUNNING Forced Alignment Experiments"
for s in ${strands[@]};
do
    #convert to 0-base indexing
    strandId=$((s-1)) 
    for window in ${window_size[@]};
    do
	real_strandId=$((strandId+1))
	fast5_path="strand_${real_strandId}_fast5"
	score_file=${PREFIX}${DENSITY}_"bonito_ctc_${sim}_strand${real_strandId}.hdf5"
	out_file=${PREFIX}${DENSITY}_"bonito_forced_${sim}_strand${real_strandId}.out"
	err_file=${PREFIX}${DENSITY}_"bonito_forced_${sim}_strand${real_strandId}.err"
	h5_file=${PREFIX}${DENSITY}_"bonito_forced_${sim}_strand${real_strandId}.hdf5"
	echo "--------------------------------------------------------------"
	echo "Strand ID ${real_strandId}"
	echo "Score Path ${score_file}"
	echo "Output File ${out_file}"
	echo "Error File ${err_file}"
	echo "Output Alignment Path ${h5_file}"
	echo "--------------------------------------------------------------" 
	sbatch --time "24:00:00"  --exclude="c21,c32,c34,c[58-59],c68" --exclusive -J ${sim}_force -o ~/${out_file} -e ~/${err_file} -n 1  -p ${GPU}   --wrap "source ~/.bashrc && conda activate bonito_cuda && python /home/kvolkel/bonito/my_tools/force_alignment_map.py -f ${H5_DIR}/${score_file} -o ${H5_DIR}/${h5_file} -s ${DNA_HOME}/${DENSITY}_strands.dna -i 204 ${strandId} 0"
    done
done


