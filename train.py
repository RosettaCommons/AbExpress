#!/usr/bin/env python3

import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

import lstm
import preprocess

def create_parser():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--msa",
        required=True,
        help="Path to the multiple sequence alignment. The sequences should be aligned by IMGT Numbering or other Ab specific numbering schema")
    parser.add_argument(
        "--msa_fmt",
        default="fasta",
        help="Chose one of the MSA formats supported by biopython")
    parser.add_argument(
        "--label", required=True,
        help="Path to training labels. One label per file. Requires as many labels as sequences in the MSA.")
    parser.add_argument(
        "--paired", action="store_true",
        help="The MSA contains heavy light chain paired sequences, whereas the light chain always follows the heavy chain in the next row")
    parser.add_argument(
        "--alphabet",
        default="cdrpeyvmtiqslkgnwahf-",
        help="The complete amino acid alphabet that's expected to be found in the sequences")
    parser.add_argument(
        "--epochs", type=int, default=100,
        help="Number of training epochs [default: %(default)s]")
    parser.add_argument(
        "--fold", type=int, default=10,
        help="Number of cross-validations [default: %(default)s]")
    parser.add_argument(
        "--loss", default="mse",
        help="Loss function [default: %(default)s]")
    parser.add_argument(
        "--dropout", default=0.1,
        type=float,
        help="[default: %(default)s]")
    parser.add_argument(
        "--shuffle",
        action="store_true",
        help="Shuffle training labels for benchmark purposes ONLY")
    parser.add_argument(
        "--cp_period",
        type=int,
        default=1,
        help="Number of epochs to validate/write checkpoints [default: %(default)s]")
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Checkpoint path prefix. Will be extended with the cross-validation runid")
    parser.add_argument(
        "--workers",
        type=int, default=2,
        help="Number of workers [default: %(default)s]")
    return parser

def crossValidate(X, Y, splits=10, shuffle=True, devsplit=None):
    from sklearn.model_selection import StratifiedKFold

    kfold = StratifiedKFold(splits, shuffle)

    for train_indices, test_indices in kfold.split(X, Y):
        if devsplit:
            devsize = int(devsplit*len(Y))
            dev_indices = train_indices[-devsize:]
            train_indices = train_indices[:-devsize]
            yield train_indices, dev_indices, test_indices
        else:
            yield train_indices, [], test_indices

def train(frame, timesteps, features, options):
    inputshape = (timesteps, features)

    nlabels = len(set(frame["label"].tolist()))
    
    print("Creating LSTM model for mapping %s->%s" % (inputshape, nlabels))

    model = lstm.buildModel_LSTM_64_16(inputshape, nlabels, options)
    model.summary()
    history = list()
    for xfold, (train, dev, test) in enumerate(crossValidate(
            frame["onehot"],
            frame["label"],
            devsplit=0.1,
            splits=options.fold)):
        checkpoint = options.checkpoint + "_" + str(xfold)

        print("Validation run %s with train/dev/test size %s/%s/%s" % (xfold, len(train), len(dev), len(test)))

        # Preparing training data
        x = list()
        for row in frame.iloc[train]["onehot"].values:
            x.append(np.hstack(list(np.hstack(sample) for sample in row)))
        x = np.array(x)
        x = x.reshape(len(train), *inputshape)
        y = tf.keras.utils.to_categorical(frame.iloc[train]["label"])

        # Preparing test data
        a = list()
        for row in frame.iloc[test]["onehot"].values:
            a.append(np.hstack(list(np.hstack(sample) for sample in row)))
        a = np.array(a)
        a = a.reshape(len(test), *inputshape)
        b = tf.keras.utils.to_categorical(frame.iloc[test]["label"])

        # Preparing dev data
        dx = list()
        for row in frame.iloc[dev]["onehot"].values:
            dx.append(np.hstack(list(np.hstack(sample) for sample in row)))
        dx = np.array(dx)
        dx = dx.reshape(len(dev), *inputshape)
        dy = tf.keras.utils.to_categorical(frame.iloc[dev]["label"])

        if options.shuffle:
            print("!> Randomizing training labels <!", args)
            np.random.shuffle(y)

        checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint, 
            verbose=1,
            monitor='val_loss',
            save_best_only=True,
            save_weights_only=True,
            mode="min",
            period=options.cp_period)

        model = lstm.buildModel_LSTM_64_16(inputshape, nlabels, options)
        fit_history = model.fit(
            x, y,
            shuffle=True,
            validation_freq=options.cp_period,
            workers=options.workers,
            validation_data=[dx, dy],
            callbacks=[checkpoint_cb],
            epochs=options.epochs)

        model.load_weights(checkpoint)
        evaluation = model.evaluate(a, b)
        loss, acc, auc, recall = evaluation
        print("Loss: %s; Accuracy: %s; Recall: %s; AUC: %s" % (loss, acc, recall, auc), xfold, options.fold)
        y_pred_keras = model.predict(a).ravel()

        samples = {
            "train": train,
            "dev": dev,
            "test": test
        }

        summary = {
            "checkpoint": checkpoint,
            "samples": samples,
            "loss": loss,
            "acc": acc,
            "AUC": auc,
            "recall": recall,
            "expected": b,
            "prediction": y_pred_keras
        }

        yield summary
            

if __name__ == "__main__":
    parser = create_parser()
    options = parser.parse_args()

    word2vec = preprocess.genWord2Vec(sorted(options.alphabet))
    data, timesteps, features = preprocess.embed_onehot(preprocess.read_sequences_and_labels(options), word2vec)
    
    history = list()
    for summary in train(data, timesteps, features, options):
        history.append(summary)

    print("Cross validation summary:")
    print(f"Samples: {len(data)}; Timesteps: {timesteps}; Features: {features}")
    for summary in history:
        msg = "Epoch summary: " + "; ".join(["%s: %s" % (kk, vv) for kk, vv in zip(summary.keys(), summary.values()) if kk in ["loss", "acc", "recall", "auc"]])
        print(msg)
