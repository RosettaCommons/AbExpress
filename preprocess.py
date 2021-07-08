#!/usr/bin/env python3

import tensorflow as tf
import numpy as np
import pandas as pd

def genWord2Vec(alphabet):
    word2vec = dict(zip(alphabet, range(len(alphabet)+1)))
    return word2vec

def read_sequences(options):
    from Bio import AlignIO
    
    alignment = AlignIO.read(options.msa, options.msa_fmt)

    if options.paired and len(alignment) % 2 != 0:
        raise Exception(f"Heavy/Light chain paired sequences. Even number of sequences expected, got {len(alignment)}")

    # read sequences + labels and concatenate if necessary
    records = list()
    for rr, seq in enumerate(alignment):
        record = {
            "id" : [seq.description],
            "sequence": [list(str(seq.seq).lower())]
        }

        if options.paired:
            if (rr+1) % 2 == 0:
                last = records[-1]
                records[-1]["id"] += record["id"]
                records[-1]["sequence"] += record["sequence"]
            else:
                records.append(record)
        else:
            records.append(record)

    return records

def read_sequences_and_labels(options):
    from Bio import AlignIO
    
    alignment = AlignIO.read(options.msa, options.msa_fmt)
    labels = pd.read_csv(options.label, skip_blank_lines=True, header=None)

    if len(alignment) != len(labels):
        raise Exception(f"Alignment length ({len(alignment)}) != Label length ({len(labels)})")

    if options.paired and len(alignment) % 2 != 0:
        raise Exception(f"Heavy/Light chain paired sequences. Even number of sequences expected, got {len(alignment)}")

    # read sequences + labels and concatenate if necessary
    records = list()
    for rr, (seq, label) in enumerate(zip(alignment, labels.iterrows())):
        record = {
            "sequence": [list(str(seq.seq).lower())],
            "label": label[1][0]
        }

        if options.paired:
            if (rr+1) % 2 == 0:
                last = records[-1]
                if last["label"] != record["label"]:
                    raise Exception(f"Labels for paired sequence {rr} and {rr+1} must match for one sequence pair. Got {last['label']} and {record['label']}")

                records[-1]["sequence"] += record["sequence"]
            else:
                records.append(record)
        else:
            records.append(record)

    return records

def embed_onehot(records, word2vec):
    timesteps = None; features = None
    
    # Create one-hot arrays
    onehot = None
    for record in records:
        features = len(word2vec) * len(record["sequence"]) # TODO: Is it helpful to check if that always returns the same number?

        record["onehot"] = list()
        for sequence in record["sequence"]:
            timesteps = len(sequence) #This should always be the MSA width
            sequence = " ".join(sequence)
            sequence = tf.keras.preprocessing.text.text_to_word_sequence(sequence, filters='')
            sequence = np.array([word2vec[word] for word in sequence])
            onehot = tf.keras.utils.to_categorical(sequence, num_classes=len(word2vec))
            record["onehot"].append(onehot)
            
    frame = pd.DataFrame(records)
    
    return frame, timesteps, features
