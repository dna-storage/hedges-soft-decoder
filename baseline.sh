#!/bin/bash

source /home/${USER}/.bashrc
module unload cuda
conda activate bonito_env

sbatch --time "24:00:00" -J baseline -o $PAUL_NANOPORE_DATA/baseline1.fastq -e $PAUL_NANOPORE_DATA/baseline1.err -N 1 -n 1 -p rtx2080 --wrap "bonito basecaller dna_r9.4.1_e8_hac@v3.3  $PAUL_NANOPORE_DATA/221118_dna_volkel_strand1_fast5_10k_subset"

sbatch --time "24:00:00" -J baseline -o $PAUL_NANOPORE_DATA/baseline2.fastq -e $PAUL_NANOPORE_DATA/baseline2.err -N 1 -n 1 -p rtx2080 --wrap "bonito basecaller dna_r9.4.1_e8_hac@v3.3  $PAUL_NANOPORE_DATA/221118_dna_volkel_strand2_fast5_10k_subset"
