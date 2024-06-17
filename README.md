[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.11454877.svg)](https://doi.org/10.5281/zenodo.11454877)

# Soft Decoding HEDGES Codes with Bonito

This repository holds the code for the bioinformatics release of the Alignment Matrix algorithm and Beam Trellis algorithm used for soft decoding CTC matrices of nanopore reads. There are two main branches for this repository: `batch` and `no-batch`. The `batch` branch implements parallelized read batching to enable higher thread occupancy of GPU devices and is the primary branch of this project. Due to the complexity of the HPC environment and dependencies, each branch is provided with a singularity definition file that will install that branch and create an image that can be used to run our commands within the container. We also include a pre-built image available at https://cloud.sylabs.io/library/kvolkel/nanopore-soft-decoders/soft-decode of our Alignment Matrix algorithm, we do not include the port of the Beam Trellis Algorithm given space limitations for hosting images. In the following steps we will show how to build the images locally or pulling the pre-built image 

## General Requirements
The following are the main requirements for this repository to function. Our image builds include these dependencies (except for CUDA drivers which will be covered later) out of the box and no further installation of dependencies is required outside of singularity itself.

* Singularity is recomended to enable usage of our built images and definition files (Tested on Singularity Version 3.7.1-5.1.ohpc.2.1)
* NVIDIA GPU devices to run the soft decoder algorithms. CPU implementation was depracated due to the extremely high overhead of decoding individual reads.
* Linux System for Singularity Containers (Rocky Linux release 8.9 (Green Obsidian) is the most recent OS tested with our implementations)
* CUDA Drivers, (tested up to Driver Version 545.23.08).
* A working installation of CUDA libraries and developer toolkits (Tested on Cuda Version 11.7, included in the image build process)
* PyTorch (Version 2.0.0, included in image)

## Building Image Locally

The following is a set of commands to build an image for a branch locally. Here we assume that the image can be built on a platform that has sudo access. The image is also set up to enable access to all installed code for all users who run the container. Typically, shared HPC systems don't allow for sudo access to users, but after building the image on a local machine you should be able to move the `.sif` file to the shared HPC encvironment and run as a user there with no needed extra privileges. 

```
git clone  https://github.com/dna-storage/bonito_hedges
cd bonito_hedges
git checkout batch #Change "batch" to "no-batch" if you want to build the beam trellis version 
sudo singularity build bonito-hedges.sif build-bonito.def
```

## Pull Image from Library
The following command pulls the iamge from Sylabs hosted libraries. No additional work should be neccessary if this approach is taken. As was mentioned before, this container only supports the fast-batched version of the Alignment Matrix algorithm.

```
singularity pull bonito-hedges.sif library://kvolkel/nanopore-soft-decoders/soft-decode:batch
```

## Starting Up Images

To run the image, we recommend running the image in shell mode as follows. Note, the `--nv` flag is necessary in order to obtain the necessary NVIDIA device driver libraries from the host system. All other necessary CUDA libraries used to develop this project are included in the image installation. So, care just needs to be taken that the CUDA libraries of the image (as noted in the General Requirements) are compatible with the drivers on the host system.

```
singularity shell --nv  bonito-hedges.sif
```

## Running Commands
Commands utilize the same general interface used for basecalling with the base bonito software. The main input of the commands are a path to the fast5 data for the reads obtained from nanopore sequencing (simulated reads are possible as well using simulators like DeepSimulator or Squigulator). Outputs are fastq formatted files representing the basecall that was decoded from the soft decoders.

### Alignment Matrix Algorithm Command 

Before running the following command, ensure that you are running an environment that supports the batched version of the Alignment Matrix Algorithm. An example of doing such is as follows:

```
singularity pull bonito-hedges.sif library://kvolkel/nanopore-soft-decoders/soft-decode:batch
singularity shell --nv  bonito-hedges.sif
```

The following command gives an overview of all relevant inputs and parameters for the soft-decoder. The first two arguments are paths to the model being used (weights and model config) and the FAST5 path of raw signals to be decoded (basecalled) respectively. The following two parameters `--disabel_half` and `--disable_koi` are included to make the bonito code base cooperate with the older CTC-based model. `--strand_pad` is the `3'` padding we force an alignment to for CTC matrix pruning, and `--hedges_bytes <byte list>` represents the index bytes that we factor out from the CTC matrix as well. `--hedges_params` is a path to the parameters describing the hedges code being used. `--trellis base` specifies that we are using a basic convolutional code trellis. `--batch_size <batch-size>` is the number of reads that will be decoded in parallel on the same GPU via batching of their matrices. `--window <window-size>` is an optional number between 0-1 that describes the percentage of time-steps that should be calculated during the forward algorithm portion of edge score calculation, NOTE: this is an experimental parameter and can degrade performance for any value <1 (e.g. using less than 100% of the total time steps available). We found that any batch size up to about 32-50 is a reasonable choice to saturate GPU scheduling resources for an RTX 2060 Super NVIDIA GPU. `--lower_index <lower-index>` and `--upper_index <upper-index>` allow for limiting the number of reads decoded in a single run, e.g. `--lower_index 0` : `--upper_index 200` will complete the first 200 reads exactly. By default, the entire FAST5 data set is decoded. 

NOTE: 
* The output of the command (the basecalls/decoded DNA strings) will be written to stdout. 
* The output only includes bases related to the HEDGES code, e.g. index and payload bytes. Auxillary regions like padding, poly-A, etc. are not output.

```
basecaller /bonito_hedges/bonito/models/dna_r9.4.1@v2  /bonito_hedges/example/221118_dna_volkel_strand1_fast5_10_debug_subset --disable_half --disable_koi --strand_pad GGCGACAGAAGAGTCAAGGTTC --hedges_bytes <byte-list> --hedges_params /bonito_hedges/example/hedges_options.json  --trellis base --batch_size <batch-size> --window <window-size> --lower_index <lower-index> --upper_index <upper-index>
```


### Beam Trellis Algorithm Command

Running the GPU version of the beam trellis algorithm is mostly the same, the only steps that need to be taken first is to start the appropriate environment by doing something like the following:

```
git clone  https://github.com/dna-storage/bonito_hedges
cd bonito_hedges
git checkout no-batch
sudo singularity build bonito-hedges.sif build-bonito.def
singularity shell --nv  bonito-hedges.sif
```

Then, the Beam Trellis can be run with the following command. This is mostly the same command, but we do not support batching for the Beam Trellis algorithm because of the already large load of threads that the Trellis places on the GPU devices. Note, the name of the trellis is changed from `base` to `beam_1` in this command. All other shown parameters retain the same meaning.

```
basecaller /bonito_hedges/bonito/models/dna_r9.4.1@v2  /bonito_hedges/example/221118_dna_volkel_strand1_fast5_10_debug_subset --disable_half --disable_koi --strand_pad GGCGACAGAAGAGTCAAGGTTC --hedges_bytes <byte-list> --hedges_params /bonito_hedges/example/hedges_options.json  --trellis beam_1 --lower_index <lower-index> --upper_index <upper-index>
```

### Test Command to Verify Installation

We include a small 10 read test FAST5 directory to ensure that the Bonito code base and our soft decoding libraries are installed properly. The following commands will start the Singularity image and then run the test data set. The expected output is a FASTQ file stream with no errors related to library imports. If you get errors that indicate CUDA libraries cannot be found, ensure that the `--nv` flag is used to start the container's execution. 

``` shell
bonito basecaller /bonito/bonito/models/dna_r9.4.1@v2  /bonito/hedges-test/221118_dna_volkel_strand1_fast5_10_debug_subset --disable_half --disable_koi --strand_pad GGCGACAGAAGAGTCAAGGTTC --hedges_bytes 204 0 --hedges_params /bonito/hedges-configs/231206_dna_datastorage_1667_batch_full_1667.json  --trellis base --batch_size 10
```


## Using Soft Decoders as a Library

After installing this package with `make develop` in a conda or native environment, or when running a container, the functions for CTC soft decoding of HEDGES can be imported to integrate with any model. That is, our algorithms and implementations are not necessarily dependent on the Bonito code base. This allows us to analyze soft decoding performance against any model that emits CTC data, and this flexibility is what allows us to evaluate the Alignment Matrix Algortihm with the RODAN RNA open source model. This can be accomplished with the following code:

``` python

import bonito.hedges_decode.hedges_decode as hd

 hedges_decode(read_id, #list of read-ids
	scores_arg, #dictionary {"scores": NxTxA tensor of float32} N - number of batched reads, T - CTC time dimension A - Alphabet size 
	hedges_params:str, #path to hedges parameters, examples can be found in the hedges-configs/ directory
	hedges_bytes:bytes, #bytearray of bytes that represent the index of the read (identical to the <hedges_bytes> parameter to the command-line basecaller argument)
	using_hedges_DNA_constraint:bool, #set whether the trellis should use constraints of HEDGES code in decoding process, set to False (feature not fully implemented yet)
	alphabet:list, #alphabet of the CTC matrx - Bonito uses the array ["N", "A", "C", "G", "T"] which represents the character at each position of the CTC matrix 
	endpoint_seq:str="", #endpoint string to force alignment for CTC trimming, indentical to the <strand_pad> command line parameter
	window=0, #window to use for approximate alignments. Defaults to 0 which means 100% of the alignment is calculated. Any number>0 and <1 is considered as the percentage of window to calculate
	trellis="base", #trellis type to use - usually this is "base"
	rna=False, #whether RNA is being decoded or not, set to True if RNA CTC data so that only forward directions are considered
	ctc_dump=None #Set to True if you want to extract the raw and pruned CTC data for a read, it is recomended to ensure your batch size is 1 in this case. Decoding only runs up to the pruning process.
	)
```

You can find an example of using the standalone function in `./debug/hedges_ctc_debug.py`. The interface design was mostly driven to easily integrate with Bonito's handling of read information. We provide utilities in `./bonito/hedges_decode/hedges_decode_utils.py` that will iteratively generate and help package individual CTC score tensors for a set of reads into a single batch-tensor that is formatted for the functions `scores_arg` argument. Our port of the RODAN model to enable soft-decoder basecalling can be found at https://github.com/dna-storage/RODAN-HEDGES.


## Development

All code related to the CTC soft decoding library can be found under the module `./bonito/hedges_decode`. Implementations of the Beam Trellis algorithm, including CUDA ports of the original algorithm, can be found at `./bonito/hedges_decode/beam_viterbi`. 

Core source code files for the Alignment Matrix Algortihm are: 
 - `./bonito/hedges_decode/hedges_decode.py` - top level functions for the soft decoding library 
 - `./bonito/hedges_decode/base_decode.py` - top level classes that implement the information flow of the HEDGES code trellis
 - `./bonito/hedges_decode/decode_ctc.py` - implementation of Alignment Matrix forward algorithm scoring mechanisms. This file calls CUDA implementations of the forward algorithm/edge score aggregation that can be found in `./bonito/hedges_decode/cuda/`.


## Analysis Notebooks and Encoded Data

Notebooks generating figures used in the **Bioinformatics** submission and instructions for running the notebooks can be found in the `/bioinformatics-analysis` directory. The raw data that was encoded for experiments, and the exact set of strands that were ordered for synthesis can be found in the `/encoded-data` directory. The synthesized strands are found in the spreadsheet `/encoded-data/synthesized-strands.xlsx`.




### References
 - [RODAN Open Source RNA Basecaller](https://github.com/biodlab/RODAN)
 - [Sequence Modeling With CTC](https://distill.pub/2017/ctc/)
 - [Quartznet: Deep Automatic Speech Recognition With 1D Time-Channel Separable Convolutions](https://arxiv.org/pdf/1910.10261.pdf)
 - [Pair consensus decoding improves accuracy of neural network basecallers for nanopore sequencing](https://www.biorxiv.org/content/10.1101/2020.02.25.956771v1.full.pdf)

### License and Copyright
(c) 2019 Oxford Nanopore Technologies Ltd.

Bonito is distributed under the terms of the Oxford Nanopore
Technologies, Ltd.  Public License, v. 1.0.  If a copy of the License
was not distributed with this file, You can obtain one at
http://nanoporetech.com

