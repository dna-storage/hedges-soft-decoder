# Soft Decoding HEDGES Codes with Bonito

This repository holds the code for the bioinformatics release of the Alignment Matrix algorithm and Beam Trellis algorithm used for soft decoding CTC matrices of nanopore reads. There are two main branches for this repository: `main` and `batch`. The `batch` branch implements parallelized read batching to enable higher thread occupancy of GPU devices. Due to the complexity of the HPC environment and dependencies, each branch is provided with a singularity definition file that will install that branch and create an image that can be used to run our commands within the container. We also include a pre-built image available at https://cloud.sylabs.io/ of our Alignment Matrix algorithm, we do not include the port of the Beam Trellis Algorithm given space limitations for hosting images. In the following steps we will show how to build the images locally or pulling the pre-built image 

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
git checkout batch #Change "batch" to "hedges_decode" if you want to build the beam trellis version 
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
git checkout hedges_decode 
sudo singularity build bonito-hedges.sif build-bonito.def
singularity shell --nv  bonito-hedges.sif
```

Then, the Beam Trellis can be run with the following command. This is mostly the same command, but we do not support batching for the Beam Trellis algorithm because of the already large load of threads that the Trellis places on the GPU devices. Note, the name of the trellis is changed from `base` to `beam_1` in this command. All other shown parameters retain the same meaning.

```
basecaller /bonito_hedges/bonito/models/dna_r9.4.1@v2  /bonito_hedges/example/221118_dna_volkel_strand1_fast5_10_debug_subset --disable_half --disable_koi --strand_pad GGCGACAGAAGAGTCAAGGTTC --hedges_bytes <byte-list> --hedges_params /bonito_hedges/example/hedges_options.json  --trellis beam_1 --lower_index <lower-index> --upper_index <upper-index>
```





### References
 - [Sequence Modeling With CTC](https://distill.pub/2017/ctc/)
 - [Quartznet: Deep Automatic Speech Recognition With 1D Time-Channel Separable Convolutions](https://arxiv.org/pdf/1910.10261.pdf)
 - [Pair consensus decoding improves accuracy of neural network basecallers for nanopore sequencing](https://www.biorxiv.org/content/10.1101/2020.02.25.956771v1.full.pdf)

### Licence and Copyright
(c) 2019 Oxford Nanopore Technologies Ltd.

Bonito is distributed under the terms of the Oxford Nanopore
Technologies, Ltd.  Public License, v. 1.0.  If a copy of the License
was not distributed with this file, You can obtain one at
http://nanoporetech.com

