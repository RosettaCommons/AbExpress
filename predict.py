#!/usr/bin/env python3

import tensorflow as tf
import numpy as np

import lstm
import preprocess

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
        help="TSV file with predictions")
    return parser

def test(data, timesteps, features, options):
    inputshape = (timesteps, features)

    nlabels = 2 # FIXME: don't hardcode

    
    model = lstm.buildModel_LSTM_64_16(inputshape, nlabels, options)
    model.load_weights(options.checkpoint)
    model.summary()
    
    x = list()
    for row in data["onehot"].values:
        x.append(np.hstack(list(np.hstack(sample) for sample in row)))
    x = np.array(x)
    x = x.reshape(len(data), *inputshape)

    print("Prediction expression class of %s entities" % (len(data)))
    prediction = model.predict(x)

    return prediction
    
if __name__ == "__main__":
    parser = create_parser()
    options = parser.parse_args()

    word2vec = preprocess.genWord2Vec(sorted(options.alphabet))
    data, timesteps, features = preprocess.embed_onehot(preprocess.read_sequences(options), word2vec)

    prediction = test(data, timesteps, features, options)

    fh = open(options.output, 'w')
    fh.write("SequenceID\tPrediction\n")
    for sample, pred in zip(data.id, prediction):
        fh.write(f'{";".join(sample)}\t{pred[1]}\n')
    fh.close()
    
