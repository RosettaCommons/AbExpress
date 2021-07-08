#!/usr/bin/env python3

import tensorflow as tf
import numpy as np
import pandas as pd

import lstm
import preprocess
import predict

def create_parser():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--msa",
        required=True,
        help="Path to the multiple sequence alignment. The sequences should be aligned by IMGT Numbering or other Ab specific numbering schema. The used numbering and msa width must match the msa used for training.")
    parser.add_argument(
        "--msa_fmt",
        default="fasta",
        help="Chose one of the MSA formats supported by biopython")
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Path to the checkpoint containing weights of a trained model")
    parser.add_argument(
        "--paired", action="store_true",
        help="The MSA contains heavy light chain paired sequences, whereas the light chain always follows the heavy chain in the next row")
    parser.add_argument(
        "--alphabet",
        default="cdrpeyvmtiqslkgnwahf-",
        help="The complete amino acid alphabet that's expected to be found in the sequences")
    parser.add_argument(
        "--loss", default="mse",
        help="Loss function [default: %(default)s]")
    parser.add_argument(
        "--dropout", default=0.1,
        type=float,
        help="[default: %(default)s]")
    parser.add_argument(
        "-o", "--output", required=True,
        help="TSV file with engineerability for the given checkpoint")
    return parser

def single_point_mutants(sequences, options):
    variants = list()
    for ss, sequence in enumerate(sequences):
        resu = 0
        for resi, reswt in enumerate(sequence):
            if reswt == "-":
                continue
            for resmt in options.alphabet:
                if resmt == "-":
                    continue

                spm = [ss.copy() for ss in sequences]
                spm[ss][resi] = resmt
                variants.append({
                    "recordi": ss,
                    "variant": (reswt, resi+1, resmt),
                    "variant_ungapped": (reswt, resu+1, resmt),
                    "sequence": spm
                })
            resu += 1
    return variants

if __name__ == "__main__":
    parser = create_parser()
    options = parser.parse_args()

    word2vec = preprocess.genWord2Vec(sorted(options.alphabet))
    msa = preprocess.read_sequences(options)


    fh = open(options.output, 'w')
    fh.write("SequenceID\tEngineerability [%]\n")
    for record in msa:
        spm = single_point_mutants(record["sequence"], options)
        data, timesteps, features = preprocess.embed_onehot(spm, word2vec)

        prediction = predict.test(data, timesteps, features, options)
        engineerability = max(prediction[:, 1]) * 100
        seqid = ";".join(record["id"])
        fh.write(f"{seqid}\t{engineerability}\n")
    fh.close()
