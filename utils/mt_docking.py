"""
Author: QHGG
Date: 2021-08-02 22:46:25
LastEditTime: 2022-08-22 15:58:57
LastEditors: QHGG
Description:
FilePath: /AlphaDrug/utils/docking.py
"""
import os
import random
import subprocess
import re
from loguru import logger
from rdkit import Chem
from rdkit.Chem import AllChem
from Bio import PDB
import json
import time
import numpy as np
from rdkit.ML.Descriptors import MoleculeDescriptors
import torch
from torch import nn


def ProteinParser(pdb_id, pdb_path):
    parser = PDB.PDBParser(PERMISSIVE=1)
    structure = parser.get_structure(pdb_id, pdb_path)
    ppb = PDB.PPBuilder()
    seq = ''
    for pp in ppb.build_peptides(structure):
        seq += pp.get_sequence()
    return seq


def CaculateAffinity(smi, file_pro_protein_list, file_pro_lig_ref_list, file_anti_protein_list, file_anti_lig_ref_list, out_path='./', prefix='', vina='smina'):
    mol = Chem.MolFromSmiles(smi)
    m2 = Chem.AddHs(mol)
    AllChem.EmbedMolecule(m2)
    m3 = Chem.RemoveHs(m2)
    file_output = os.path.join(out_path, prefix + str(time.time()) + str(random.random()) + '.pdb')
    Chem.MolToPDBFile(m3, file_output)

    # mol = Chem.MolFromPDBFile("test.pdb")
    # smile = Chem.MolToSmiles(mol)
    # logger.info(smile)
    # logger.info(smi)
    pro_affinity_list = [500] * len(file_pro_protein_list)
    anti_affinity_list = [500] * len(file_anti_protein_list)
    try:
        if vina == 'smina':
            # file_drug="sdf_ligand_"+str(pdb_id)+str(i)+".sdf"
            launch_args = []
            smina_cmd_output = os.path.join(out_path, prefix + str(time.time()))

            # for i_task, file_protein, file_lig_ref in zip(range(len(file_pro_protein_list)), file_pro_protein_list, file_pro_lig_ref_list):
            #     launch_args += ["smina", "-r", file_protein, "-l", file_output, "--autobox_ligand", file_lig_ref, "--autobox_add", "10", "--seed", "1000", "--exhaustiveness", "9", ">>", smina_cmd_output + 'pro' + str(i_task), '&']
            # for i_task, file_protein, file_lig_ref in zip(range(len(file_anti_protein_list)), file_anti_protein_list, file_anti_lig_ref_list):
            #     launch_args += ["smina", "-r", file_protein, "-l", file_output, "--autobox_ligand", file_lig_ref, "--autobox_add", "10", "--seed", "1000", "--exhaustiveness", "9", ">>", smina_cmd_output + 'anti' + str(i_task), '&']

            for i_task, file_protein, file_lig_ref in zip(range(len(file_pro_protein_list)), file_pro_protein_list,
                                                          file_pro_lig_ref_list):
                launch_args += ["smina", "-r", file_protein, "-l", file_output, "--autobox_ligand", file_lig_ref,
                                "--autobox_add", "10", "--seed", "1000", "--exhaustiveness", "9", "1>",
                                smina_cmd_output + 'pro' + str(i_task), "2>", "/dev/null" '&']
            for i_task, file_protein, file_lig_ref in zip(range(len(file_anti_protein_list)), file_anti_protein_list,
                                                          file_anti_lig_ref_list):
                launch_args += ["smina", "-r", file_protein, "-l", file_output, "--autobox_ligand", file_lig_ref,
                                "--autobox_add", "10", "--seed", "1000", "--exhaustiveness", "9", "1>",
                                smina_cmd_output + 'anti' + str(i_task), "2>", "/dev/null", '&']

            launch_string = ' '.join(launch_args)
            # print(launch_string)

            p = subprocess.Popen(launch_string, shell=True, stdout=subprocess.PIPE)
            p.communicate(timeout=60)

            time_count = 0
            time_out = 300
            for i_task in range(len(file_pro_protein_list)):
                # affinity = 500
                flag = False
                while not os.path.exists(smina_cmd_output + 'pro' + str(i_task)) and time_count <= time_out:
                    time.sleep(1)
                    time_count += 1
                while not flag and time_count <= time_out:
                    with open(smina_cmd_output + 'pro' + str(i_task), 'r') as f_s:
                        if 'Loop time' in f_s.read():
                            flag = True
                    time.sleep(1)
                    time_count += 1
                with open(smina_cmd_output + 'pro' + str(i_task), 'r') as f_s:
                    for lines in f_s.readlines():
                        lines = lines.split()
                        if len(lines) == 4 and lines[0] == '1':
                            affinity = float(lines[1])
                            pro_affinity_list[i_task] = affinity

            for i_task in range(len(file_anti_protein_list)):
                # affinity = 500
                flag = False
                while not os.path.exists(smina_cmd_output + 'anti' + str(i_task)) and time_count <= time_out:
                    time.sleep(1)
                    time_count += 1
                while not flag and time_count <= time_out:
                    with open(smina_cmd_output + 'anti' + str(i_task), 'r') as f_s:
                        if 'Loop time' in f_s.read():
                            flag = True
                    time.sleep(1)
                    time_count += 1
                with open(smina_cmd_output + 'anti' + str(i_task), 'r') as f_s:
                    for lines in f_s.readlines():
                        lines = lines.split()
                        if len(lines) == 4 and lines[0] == '1':
                            affinity = float(lines[1])
                            anti_affinity_list[i_task] = affinity

            p = subprocess.Popen('rm -rf ' + smina_cmd_output + '*', shell=True, stdout=subprocess.PIPE)
            p.communicate()
            p = subprocess.Popen('rm -rf ' + file_output, shell=True, stdout=subprocess.PIPE)
            p.communicate()
    except:
        pro_affinity_list = [500] * len(file_pro_protein_list)
        anti_affinity_list = [500] * len(file_anti_protein_list)
    return pro_affinity_list, anti_affinity_list


@torch.no_grad()
def sample(model, path, vocabulary, proVoc, smiMaxLen, proMaxLen, device, sampleTimes, protein_seq_list):
    model.eval()
    proba_list = []
    for protein_seq in protein_seq_list:
        pathList = path[:]
        length = len(pathList)
        pathList.extend(['^'] * (smiMaxLen - length))

        protein = '&' + protein_seq + '$'
        proList = list(protein)
        lp = len(protein)

        proList.extend(['^'] * (proMaxLen - lp))

        proteinInput = [proVoc.index(pro) for pro in proList]
        currentInput = [vocabulary.index(smi) for smi in pathList]

        src = torch.as_tensor([proteinInput]).to(device)
        tgt = torch.as_tensor([currentInput]).to(device)

        smiMask = [1] * length + [0] * (smiMaxLen - length)
        smiMask = torch.as_tensor([smiMask]).to(device)
        proMask = [1] * lp + [0] * (proMaxLen - lp)
        proMask = torch.as_tensor([proMask]).to(device)

        # tgt_mask = nn.Transformer.generate_square_subsequent_mask(model, smiMaxLen).tolist()  # pytorch 1.4
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(sz=smiMaxLen).tolist()  # pytorch 1.13
        tgt_mask = [tgt_mask] * 1
        tgt_mask = torch.as_tensor(tgt_mask).to(device)

        sl = length - 1
        out = model(src, tgt, smiMask, proMask, tgt_mask)[:, sl, :]
        out = out.tolist()[0]
        pr = np.exp(out) / np.sum(np.exp(out))
        proba_list.append(pr)

        # print(atomListExpanded)
        # print(logpListExpanded)

    pr_mean = np.asarray(proba_list).mean(axis=0)
    prList = np.random.multinomial(1, pr_mean, sampleTimes)

    indices = list(set(np.argmax(prList, axis=1)))

    atomList = [vocabulary[i] for i in indices]
    logpList = [np.log(pr_mean[i] + 1e-10) for i in indices]

    atomListExpanded = []
    logpListExpanded = []
    for idx, atom in enumerate(atomList):
        if atom == '&' or atom == '^':
            continue
        atomListExpanded.append(atom)
        logpListExpanded.append(logpList[idx])
    # logger.info(atomListExpanded)
    return atomListExpanded, logpListExpanded
