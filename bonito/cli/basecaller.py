"""
Bonito Basecaller
"""

import os
import sys
import numpy as np
from tqdm import tqdm
from time import perf_counter
from functools import partial
from datetime import timedelta
from itertools import islice as take
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

from bonito.aligner import align_map, Aligner
from bonito.reader import read_chunks, Reader
from bonito.io import CTCWriter,Writer, biofmt
from bonito.mod_util import call_mods, load_mods_model
from bonito.cli.download import File, models, __models__
from bonito.multiprocessing import process_cancel, process_itemmap
from bonito.util import column_to_set, load_symbol, load_model, init
import multiprocessing

def main(args):
    multiprocessing.set_start_method('spawn')
    init(args.seed, args.device)

    try:
        reader = Reader(args.reads_directory, args.recursive)
        sys.stderr.write("> reading %s\n" % reader.fmt)
    except FileNotFoundError:
        sys.stderr.write("> error: no suitable files found in %s\n" % args.reads_directory)
        exit(1)

    fmt = biofmt(aligned=args.reference is not None)

    if args.reference and args.reference.endswith(".mmi") and fmt.name == "cram":
        sys.stderr.write("> error: reference cannot be a .mmi when outputting cram\n")
        exit(1)
    elif args.reference and fmt.name == "fastq":
        sys.stderr.write(f"> warning: did you really want {fmt.aligned} {fmt.name}?\n")
    else:
        sys.stderr.write(f"> outputting {fmt.aligned} {fmt.name}\n")

    if args.model_directory in models and args.model_directory not in os.listdir(__models__):
        sys.stderr.write("> downloading model\n")
        File(__models__, models[args.model_directory]).download()

    sys.stderr.write(f"> loading model {args.model_directory}\n")
    try:
        model = load_model(
            args.model_directory,
            args.device,
            weights=args.weights if args.weights > 0 else None,
            chunksize=args.chunksize,
            overlap=args.overlap,
            batchsize=args.batchsize,
            quantize=args.quantize,
            use_koi=(not args.disable_koi),
            half=(not args.disable_half)
        )
    except FileNotFoundError:
        sys.stderr.write(f"> error: failed to load {args.model_directory}\n")
        sys.stderr.write(f"> available models:\n")
        for model in sorted(models): sys.stderr.write(f" - {model}\n")
        exit(1)

    if args.verbose:
        sys.stderr.write(f"> model basecaller params: {model.config['basecaller']}\n")

    basecall = load_symbol(args.model_directory, "basecall")

    mods_model = None
    if args.modified_base_model is not None or args.modified_bases is not None:
        sys.stderr.write("> loading modified base model\n")
        mods_model = load_mods_model(
            args.modified_bases, args.model_directory, args.modified_base_model,
            device=args.modified_device,
        )
        sys.stderr.write(f"> {mods_model[1]['alphabet_str']}\n")

    if args.reference:
        sys.stderr.write("> loading reference\n")
        aligner = Aligner(args.reference, preset='map-ont', best_n=1)
        if not aligner:
            sys.stderr.write("> failed to load/build index\n")
            exit(1)
    else:
        aligner = None

    if args.save_ctc and not args.reference:
        sys.stderr.write("> a reference is needed to output ctc training data\n")
        exit(1)

    if fmt.name != 'fastq':
        groups, num_reads = reader.get_read_groups(
            args.reads_directory, args.model_directory,
            n_proc=8, recursive=args.recursive,
            read_ids=column_to_set(args.read_ids), skip=args.skip,
            cancel=process_cancel()
        )
    else:
        groups = []
        num_reads = None

    reads = reader.get_reads(
        args.reads_directory, n_proc=8, recursive=args.recursive,
        read_ids=column_to_set(args.read_ids), skip=args.skip,
        do_trim=not args.no_trim, norm_params=model.config.get('normalisation'),
        cancel=process_cancel(),
        lower_index=args.lower_index,
        upper_index=args.upper_index
    )

    if args.max_reads:
        reads = take(reads, args.max_reads)

    if args.save_ctc:
        reads = (
            chunk for read in reads
            for chunk in read_chunks(
                read,
                chunksize=model.config["basecaller"]["chunksize"],
                overlap=model.config["basecaller"]["overlap"]
            )
        )
        ResultsWriter = CTCWriter
    else:
        ResultsWriter = Writer

    results = basecall(
        model, reads, reverse=args.revcomp,
        batchsize=model.config["basecaller"]["batchsize"],
        chunksize=model.config["basecaller"]["chunksize"],
        overlap=model.config["basecaller"]["overlap"],
        hedges_params=args.hedges_params,
        hedges_bytes=args.hedges_bytes,
        hedges_using_DNA_constraint=args.hedges_use_dna_constraint,
        strand_pad=args.strand_pad,
        rna=args.rna,
        window=args.window,
        trellis=args.trellis,
        mod_states=args.mod_states,
        processes=args.processes,
        batch_size=args.batch_size,
        ctc_dump=args.ctc_fast5
    )
    
    if mods_model is not None:
        if args.modified_device:
            results = ((k, call_mods(mods_model, k, v)) for k, v in results)
        else:
            results = process_itemmap(
                partial(call_mods, mods_model), results, n_proc=args.modified_procs
            )
    if aligner:
        results = align_map(aligner, results, n_thread=args.alignment_threads)

    writer_kwargs = {'aligner': aligner,
                     'group_key': args.model_directory,
                     'ref_fn': args.reference,
                     'groups': groups,
                     'min_qscore': args.min_qscore}
    if args.save_ctc:
        writer_kwargs['rna'] = args.rna
        writer_kwargs['min_accuracy'] = args.min_accuracy_save_ctc
        
    writer = ResultsWriter(
        fmt.mode, tqdm(results, desc="> calling", unit=" reads", leave=False,
                       total=num_reads, smoothing=0, ascii=True, ncols=100),
        **writer_kwargs)

    t0 = perf_counter()
    writer.start()
    writer.join()
    duration = perf_counter() - t0
    num_samples = sum(num_samples for read_id, num_samples in writer.log)

    sys.stderr.write("> completed reads: %s\n" % len(writer.log))
    sys.stderr.write("> duration: %s\n" % timedelta(seconds=np.round(duration)))
    sys.stderr.write("> samples per second %.1E\n" % (num_samples / duration))
    sys.stderr.write("> done\n")


def argparser():
    parser = ArgumentParser(
        formatter_class=ArgumentDefaultsHelpFormatter,
        add_help=False
    )
    parser.add_argument("model_directory")
    parser.add_argument("reads_directory")
    parser.add_argument("--reference")
    parser.add_argument("--modified-bases", nargs="+")
    parser.add_argument("--modified-base-model")
    parser.add_argument("--modified-procs", default=8, type=int)
    parser.add_argument("--modified-device", default=None)
    parser.add_argument("--read-ids")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", default=25, type=int)
    parser.add_argument("--weights", default=0, type=int)
    parser.add_argument("--skip", action="store_true", default=False)
    parser.add_argument("--no-trim", action="store_true", default=False)
    parser.add_argument("--save-ctc", action="store_true", default=False)
    parser.add_argument("--revcomp", action="store_true", default=False)
    parser.add_argument("--rna", action="store_true", default=False)
    parser.add_argument("--recursive", action="store_true", default=False)
    quant_parser = parser.add_mutually_exclusive_group(required=False)
    quant_parser.add_argument("--quantize", dest="quantize", action="store_true")
    quant_parser.add_argument("--no-quantize", dest="quantize", action="store_false")
    parser.set_defaults(quantize=None)
    parser.add_argument("--overlap", default=None, type=int)
    parser.add_argument("--chunksize", default=None, type=int)
    parser.add_argument("--batchsize", default=None, type=int)
    parser.add_argument("--max-reads", default=0, type=int)
    parser.add_argument("--min-qscore", default=0, type=int)
    parser.add_argument("--min-accuracy-save-ctc", default=0.99, type=float)
    parser.add_argument("--alignment-threads", default=8, type=int)
    parser.add_argument('-v', '--verbose', action='count', default=0)

    #arguments to be used with hedges basecalling
    parser.add_argument("--hedges_params",default=None,help="Path to json file describing hedges parameter")
    parser.add_argument("--hedges_bytes",type=int,default=None,nargs="+",help="Bytes to fastforward hedges state to")
    parser.add_argument("--hedges_use_dna_constraint",default=False,action="store_true",help = "Include this flag to include DNA Constraint information in hedges trellis")
    parser.add_argument("--strand_pad",action="store",default="",help="Optional padding strand that will be aligned to trim score endpoints")
    parser.add_argument("--disable_koi",default=False,action="store_true",help="Use this flag when using CTC-based model to avoid errors")
    parser.add_argument("--disable_half",action="store_true",default=False,help="Disables half precision on the model ")
    parser.add_argument("--window",action="store",type=float,default=0,help="window to use for ctc decoding")
    parser.add_argument("--trellis",action="store",type=str,default="base",help="trellis type to use")
    parser.add_argument("--mod_states",action="store",type=int,default=3,help="number of states per history")
    parser.add_argument("--lower_index",action="store",type=int,default=0,help="Index to start for basecalling in the data set, INCLUSIVE")
    parser.add_argument("--upper_index",action="store",type=int,default=10**9,help="Index to stop for basecalling, NOT_INCLUSIVE")
    parser.add_argument("--processes",action="store",type=int,default=1,help="processes to use for parallelizing basecalling")
    parser.add_argument("--batch_size",action="store",type=int,default=1,help="number of strands to batch for basecalling")
    parser.add_argument("--ctc_fast5",action="store",type=str,default=None,help="filename for dumping ctc data from the decoder")
    return parser
