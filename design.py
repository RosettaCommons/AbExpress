#!/usr/bin/env python3

import os
import re
import traceback

import numpy as np

import pyrosetta

import engineerability
import predict
import preprocess

def create_parser():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pdb", required=True, 
        help="Path to antibody PDB")
    parser.add_argument(
        "--xml", required=True,
        help="Rosetta Scripts XML file")
    parser.add_argument(
        "--script_vars",
        nargs="+",
        default=[],
        help="?")
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
        "--rosetta", nargs=argparse.REMAINDER,
        help="Additional arguments will be passed to rosetta")
    return parser

class DeepExpressabilityMoverCreator(pyrosetta.rosetta.protocols.moves.MoverCreator):
    instances_ = list()

    def __init__(self, args):
        pyrosetta.rosetta.protocols.moves.MoverCreator.__init__(self)
        self.args_ = args

    def create_mover(self):
        mover = DeepExpressabilityMover(self.args_)
        self.instances_.append(mover)
        return mover

    def keyname(self):
        return DeepExpressabilityMover.class_name()

    def provide_xml_schema(self, xsd):
        DeepExpressabilityMover.provide_xml_schema(xsd)

class DeepExpressabilityMover(pyrosetta.rosetta.protocols.moves.Mover):
    clones_ = list()
    
    def __init__(self, args):
        pyrosetta.rosetta.protocols.moves.Mover.__init__(self)

        self.args_ = args
        self.checkpoint_ = None
        self.weight_ = None
        self.percentile_ = 90
        self.scoretable_ = None
        self.my_cst_ = None

    def apply(self, pose):
        n_cst = pose.constraint_set().n_sequence_constraints()
        self.args_.msa = self.msa_
        self.args_.msa_fmt = "fasta" #TODO: tag
        self.args_.paired = True #TODO: tag
        self.args_.checkpoint = self.checkpoint_
        word2vec = preprocess.genWord2Vec(sorted(self.args_.alphabet))
        msa = preprocess.read_sequences(self.args_)

        scoretable = dict()
        for record in msa:
            spm = engineerability.single_point_mutants(record["sequence"], self.args_)
            spm_data, timesteps, features = preprocess.embed_onehot(spm, word2vec)
            pred_spm = predict.test(spm_data, timesteps, features, self.args_)
            wt_data, timesteps, features = preprocess.embed_onehot([record], word2vec)
            pred_wt = predict.test(wt_data, timesteps, features, self.args_)

            # Mapping the MSA onto the pose
            posenums = list()
            for seqid, sequence in zip(record["id"], record["sequence"]):
                ungapped = "".join(filter(lambda resn : resn != '-', sequence))
                loc = [pos for pos in re.finditer(ungapped, pose.sequence().lower())]
                if len(loc) > 1:
                    raise Exception(f"{seqid}: MSA sequence occurs multiple times in the pose. Not supported")
                elif len(loc) == 1:
                    posenums.append(loc[0].span())
                else:
                    raise Exception(f"{seqid}: MSA sequence not found in pose")

            # Calculate the scores using predictions and store them in a format that allows easy access via seqpos and residuetype
            scoretable = self.calculate_scoretable(scoretable, record, posenums, spm, pred_spm, pred_wt)

        self.scoretable_ = scoretable
        cst = DeepExpressabilityEnergyConstraint(self.scoretable_)

        if not self.constraint():
            pose.add_constraint(cst)
            self.my_cst_ = cst

    def calculate_scoretable(self, scoretable, record, posenums, spm, predictions_spm, predictions_wt):
        self.spm_scores_ = dict()

        spm_scaled_preds = dict()

        # Walk over all the single point mutants and its predicted expressability and convert it into an energy term
        for variant, pred in zip(spm, predictions_spm):
            # Which sequence of the sequence pair are we referring to
            recordi = variant["recordi"]
            
            wt_seq = record["sequence"][recordi]
            posenum = posenums[recordi]
            # Last column of the prediction refers to the true label (does express); For wild-type prediction, we always have only one result, because here we are only using paired sequences
            pred_wt = predictions_wt[0][-1]
            pred = pred[-1]

            # Scaling as described in the supplement to help avoiding local minima
            pos_scale = 1/(1-pred_wt)
            neg_scale = 1/pred_wt

            # Where does the (ungapped) msa alignment correspond to in the pose
            start, stop = posenum[0], posenum[1]
            
            spm_scaled_preds[(start, stop)] = spm_scaled_preds.get((start, stop), dict())
            scaled = (pred - pred_wt)
            scaled = (scaled/pos_scale) if scaled >= 0 else (scaled/neg_scale)

            # What mutated to what
            mutation = variant["variant_ungapped"]
            res_wt = mutation[0]; pos_mt = mutation[1]; res_mt = mutation[2]

            #self.spm_scores_[(pos_mt+posenum, res_mt)] = scaled
            spm_scaled_preds[(start, stop)][(pos_mt+start, res_mt)] = scaled
            # WT score
            spm_scaled_preds[(start, stop)][(pos_mt+start, res_wt)] = 0
            
        percentiles = dict()
        for (start, stop), _ in spm_scaled_preds.items():
            pN = np.percentile(list(_.values()), self.percentile_)
            perc = 1. / pN
            for (seqpos, resn), scaled in _.items():
                score = - (scaled * perc * self.weight_)
                score = max(-self.weight_, min(score, self.weight_))
                #print(f"{seqpos}{resn.upper()}: {score} kcal/mol")
                key = (seqpos, resn)
                if key in scoretable:
                    raise Exception(f"{key}: Multiple scores for same position")
                scoretable[key] = score
        return scoretable

    def constraint(self):
        return self.my_cst_

    def parse_my_tag(self, tag, data):
        from contextlib import suppress

        tag.setAccessed("checkpoint")
        tag.setAccessed("msa")
        tag.setAccessed("offset")
        tag.setAccessed("percentile")
        tag.setAccessed("weight")

        options = tag.getOptions()

        with suppress(KeyError):
            self.msa_ = options["msa"]
        with suppress(KeyError):
            self.checkpoint_ = options["checkpoint"]
        with suppress(KeyError):
            self.weight_ = float(options["weight"])
        with suppress(KeyError):
            self.percentile_ = float(options["percentile"])

        if not os.path.exists(self.msa_):
            raise Exception(f"Invalid path to MSA: '{self.msa_}'")

    def clone(self):
        clone = DeepExpressabilityMover(self.args_)
        clone.checkpoint_ = self.checkpoint_
        clone.msa_ = self.msa_
        clone.weight_ = self.weight_
        clone.percentile_ = self.percentile_
        clone.scoretable_ = self.scoretable_ 
        DeepExpressabilityMover.clones_.append(clone)
        return clone

    def get_name(self):
        return self.class_name()

    @staticmethod
    def class_name():
        return "DeepExpressabilityMover"

    @staticmethod
    def provide_xml_schema(xsd):
        from pyrosetta.rosetta.utility.tag import XMLSchemaAttribute, XMLSchemaType
        from pyrosetta.rosetta.utility.tag import xsct_real, xsct_positive_integer, xsct_string_cslist, xs_string, xs_boolean
        attrlist = pyrosetta.rosetta.std.list_utility_tag_XMLSchemaAttribute_t()

        attrlist.append(XMLSchemaAttribute.required_attribute(
            "msa",
            XMLSchemaType(xs_string),
            "Path to the aligned Ab Fv region"))
        attrlist.append(XMLSchemaAttribute.required_attribute(
            "checkpoint",
            XMLSchemaType(xs_string),
            "Path to the TensorFlow checkpoint containing weights of a pre-trained model"))
        attrlist.append(XMLSchemaAttribute.attribute_w_default(
            "percentile",
            XMLSchemaType(xsct_real),
            "Percentile of SPM scores to be scaled and cropped",
            "90"))
        attrlist.append(XMLSchemaAttribute.attribute_w_default(
            "weight",
            XMLSchemaType(xsct_real),
            "Weighting of the mutability bonus",
            "1"))
        
        description = '''
Uses TensorFlow to predict the Expressability of a paired heavy and light sequence. The prediction is made using a global term as well as position as well as position specific penalties to direct the mutation to certain areas in the sequence.
        '''

        pyrosetta.rosetta.protocols.moves.xsd_type_definition_w_attributes(
            xsd,
            DeepExpressabilityMover.class_name(),
            description, attrlist)

class DeepExpressabilityEnergyCreator(pyrosetta.rosetta.core.scoring.methods.EnergyMethodCreator):
    methods_ = []

    def __init__(self):
        pyrosetta.rosetta.core.scoring.methods.EnergyMethodCreator.__init__(self)
    
    def create_energy_method(self, options):
        method = DeepExpressabilityEnergy()
        DeepExpressabilityEnergyCreator.methods_.append(method)
        return method

    def score_types_for_method(self):
        stfm = pyrosetta.rosetta.utility.vector1_core_scoring_ScoreType()
        stfm.append(pyrosetta.rosetta.core.scoring.PyRosettaEnergy_last)
        return stfm

class DeepExpressabilityEnergyConstraint(pyrosetta.rosetta.core.scoring.aa_composition_energy.SequenceConstraint):
    clones_ = list()

    def __init__(self, scoretable):
        pyrosetta.rosetta.core.scoring.aa_composition_energy.SequenceConstraint.__init__(self, pyrosetta.rosetta.core.scoring.PyRosettaEnergy_last)
        self.scoretable_ = scoretable
        self.mutability_vector_ = list()

    def clone(self):
        cst = DeepExpressabilityEnergyConstraint(
            self.scoretable_)
        cst.mutability_vector_ = self.mutability_vector_
        DeepExpressabilityEnergyConstraint.clones_.append(cst)
        return cst

class DeepExpressabilityEnergy(
        pyrosetta.rosetta.core.scoring.methods.ContextIndependentOneBodyEnergy):
    clones_ = list()
    
    def __init__(self):
        pyrosetta.rosetta.core.scoring.methods.ContextIndependentOneBodyEnergy.__init__(self, DeepExpressabilityEnergyCreator())

    def indicate_required_context_graphs(self, v):
        pass

    def residue_energy(self, rsd, pose, emap):
        score_type = pyrosetta.rosetta.core.scoring.PyRosettaEnergy_last

        resn = rsd.name1()
        resi = rsd.seqpos()

        n_cst = pose.constraint_set().n_sequence_constraints()
        for csti in range(1, n_cst+1):
            cst = pose.constraint_set().sequence_constraint(csti)
            if isinstance(cst, DeepExpressabilityEnergyConstraint):
                score = cst.scoretable_.get((resi, resn.lower()))
                if score is not None:
                    score = emap.get(score_type) + score
                    print(f"{resi}{resn}: {score} kcal/mol")
                    emap.set(score_type, score)
                
    def clone(self):
        clone = DeepExpressabilityEnergy()
        self.clones_.append(clone)
        return clone

    def verison(self):
        return 1

def register_creators(options):
    energy_creator = DeepExpressabilityEnergyCreator()
    mover_creator = DeepExpressabilityMoverCreator(options)
    
    mover_factory = MoverFactory.get_instance()
    mover_factory.factory_register(mover_creator)
    PyEnergyMethodRegistrator(energy_creator)

    return mover_creator, energy_creator

def create_protocol(xmlfname, script_vars, pdbpath):
    scripts = pyrosetta.rosetta.protocols.rosetta_scripts.RosettaScriptsParser()
    
    stl_script_vars = pyrosetta.rosetta.utility.vector1_std_string()
    for var in script_vars:
        stl_script_vars.append(var)
    
    options = pyrosetta.rosetta.basic.options.process()
    pose = pyrosetta.pose_from_file(pdbpath)
    protocol = scripts.generate_mover(
        xmlfname,
        stl_script_vars)
    return pose, protocol

if __name__ == "__main__":
    from pyrosetta.rosetta.core.scoring.methods import PyEnergyMethodRegistrator
    from pyrosetta.rosetta.protocols.moves import MoverFactory

    parser = create_parser()
    options = parser.parse_args()

    pyrosetta.init(" ".join(map(str, options.rosetta)))
    creator_container = register_creators(options)

    try:
        pose, protocol = create_protocol(options.xml, options.script_vars, options.pdb)
        workpose = pose.clone()
        protocol.apply(workpose)
        protocol.final_scorefxn()(workpose)
        outfile = os.path.splitext(options.pdb)[0] + "AbExpress.pdb"
        pyrosetta.dump_pdb(workpose, outfile)
    except Exception as excptn:
        print(f'[!!] Error: {excptn}')
        traceback.print_exc()
