import torch
import time
import os
import shutil
import numpy as np
import random as rd
import argparse
from loguru import logger
import sys
from rdkit import Chem
from rdkit.Chem.QED import qed
from rdkit.Chem.Crippen import MolLogP
from rdkit.Chem import RDConfig
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')
sys.path.append(os.path.join(RDConfig.RDContribDir, 'SA_Score'))
sys.path.append(os.path.join(RDConfig.RDContribDir, 'NP_Score'))
# from rdkit.Contrib.SA_Score import sascorer
# from rdkit.Contrib.NP_Score import npscorer
import sascorer
import npscorer

from model.Lmser_Transformerr import MFT as DrugTransformer
# from model.Transformer import MFT as DrugTransformer
# from model.Transformer_Encoder import MFT as DrugTransformer

from utils.docking import CaculateAffinity, ProteinParser
from utils.log import timeLable, readSettings, saveMCTSRes
from beamsearch import sample

import pickle as pkl


infoma = {}  # used to store the docking information
moleculeParetoFront = {}  # used to store the docking information, score_vector = [binding, qed, LogP, SA, NP-likeness]
SCOREDIM = 5
REWARDMIN = np.zeros(SCOREDIM)


class Node:
    def __init__(self, parentNode=None, childNodes=[], path=[], p=1.0, smiMaxLen=999):
        self.parentNode = parentNode
        self.childNodes = childNodes
        self.wins = np.zeros(SCOREDIM)
        self.visits = 0
        self.path = path  # MCTS 路径
        self.p = p
        self.smiMaxLen = smiMaxLen

    def SelectNode(self):
        # select a child node by the child node values of PUCT algorithm
        nodeStatus = self.getExpandStatus()
        if nodeStatus == 4:  # 4: legal non-leaf node
            puct = []
            # print("child number is {}".format(len(self.childNodes)))
            for childNode in self.childNodes:
                puct.append(childNode.CaculatePUCT())
            # compute the Pareto Front given statistics of all child nodes
            indices = getParetoFrontNodeIndices(puct)
            ind = rd.choice(indices)
            return self.childNodes[ind], self.childNodes[ind].getExpandStatus()

        return self, nodeStatus

    def AddNode(self, content, p):
        n = Node(self, [], self.path + [content], p=p, smiMaxLen=self.smiMaxLen)
        self.childNodes.append(n)
        return n

    def UpdateNode(self, wins):
        self.visits += 1
        self.wins += wins

    def CaculatePUCT(self):
        if not self.parentNode:
            return 0.0  # 画图用的
        c = 1.5
        if self.visits == 0:
            wins = REWARDMIN
        else:
            wins = self.wins / self.visits  # a vector
        return wins + c * self.p * np.sqrt(self.parentNode.visits) / (1 + self.visits)

    def getExpandStatus(self):
        """
            node status: 1 terminal; 2 too long; 3 legal leaf node; 4 legal non-leaf node
        """
        if self.path[-1] == '$':
            return 1
        elif not (len(self.path) < self.smiMaxLen):
            return 2
        elif len(self.childNodes) == 0:
            return 3
        return 4


def getParetoFrontNodeIndices(puct_vectors):
    indices = []
    if len(puct_vectors) == 1:
        indices.append(0)
        return indices
    for i in range(len(puct_vectors)):
        puct_vector = np.array(puct_vectors[i])
        # [0.5, 1, 0.5] should be dominated by [0.6, 1, 0.6]
        if max((puct_vector <= np.array(puct_vectors[:i] + puct_vectors[i + 1:])).mean(axis=1)) < 1:  # non-dominated
            indices.append(i)
    return indices


def getScoreVector(mol_smiles, score_vector):
    global moleculeParetoFront
    if len(moleculeParetoFront) == 0:
        reward_vector = np.ones(SCOREDIM)
    else:
        reward_vector = np.zeros(SCOREDIM)
        for k, v in moleculeParetoFront.items():
            # if one metric is equal, should be positive
            reward_vector += score_vector >= np.asarray(v)
        reward_vector = reward_vector / len(moleculeParetoFront)
    return reward_vector


def updateParetoFront(mol_smiles, score_vector):
    global moleculeParetoFront

    if len(moleculeParetoFront) == 0:
        moleculeParetoFront[mol_smiles] = score_vector
    else:
        dominate_flag = False
        non_dominate_flag = True
        deleted_molecules = []
        for k, v in moleculeParetoFront.items():
            # [0.6, 1, 0.6] should be better than [0.5, 1, 0.5]
            if (score_vector >= np.asarray(v)).all():  # update new Pareto non-dominated point and remove invalid ones
                deleted_molecules.append(k)
                dominate_flag = True
            # [0.5, 1, 0.5] should not be added when compared with [0.6, 1, 0.6]
            if not (score_vector > np.asarray(v)).any():
                non_dominate_flag = False
        if dominate_flag or non_dominate_flag:
            moleculeParetoFront[mol_smiles] = score_vector
        for k in deleted_molecules:
            del moleculeParetoFront[k]


def IsPathEnd(path, smiMaxLen):
    # check whether the inputted path is the ending
    return (path[-1] == '$') or (len(path) >= smiMaxLen)


def Select(rootNode):
    while True:
        rootNode, nodeStatus = rootNode.SelectNode()
        if nodeStatus != 4:  # 4: legal non-leaf node
            return rootNode, nodeStatus


def Expand(rootNode, atomList, plist):
    if not IsPathEnd(rootNode.path, rootNode.smiMaxLen):
        for i, atom in enumerate(atomList):
            rootNode.AddNode(atom, plist[i])


def Update(node, wins):
    # update node values along the path from bottom to top
    while node:
        node.UpdateNode(wins)
        node = node.parentNode


def rollout(node, model):
    path = node.path[:]
    smiMaxLen = node.smiMaxLen

    thisScore = []
    thisReward = []
    thisValidSmiles = []
    thisSmiles = []

    while not IsPathEnd(path, smiMaxLen):
        # 快速走子, calculate the probabilities of next child node atoms of the current node
        atomListExpanded, pListExpanded = sample(model, path, vocabulary, proVoc, smiMaxLen, proMaxLen, device, 30, protein_seq)
        m = np.max(pListExpanded)
        indices = np.nonzero(pListExpanded == m)[0]
        ind = rd.choice(indices)
        path.append(atomListExpanded[ind])

    if path[-1] == '$':
        smileK = ''.join(path[1:-1])
        thisSmiles.append(smileK)

        mols = Chem.MolFromSmiles(smileK)  # a Mol object, None on failure.

        if mols and len(smileK) < smiMaxLen:  # valid molecule
            global infoma
            global moleculeParetoFront
            if smileK in infoma:
                score_vector = infoma[smileK]
                affinity = -score_vector[0]
                reward_vector = getScoreVector(smileK, score_vector)
            else:
                if args.docking:
                    affinity = CaculateAffinity(smileK, file_protein=pro_file[args.k], file_lig_ref=ligand_file[args.k], out_path=resFolderPath)  # - qed(Chem.MolFromSmiles(smileK))
                else:
                    affinity = 0

                score_vector = [-affinity,
                                qed(mols),
                                1 if -0.4 < MolLogP(mols) < 5.6 else 0,  # Ghose filter
                                -sascorer.calculateScore(mols),
                                npscorer.scoreMol(mols, fscore),
                                ]
                infoma[smileK] = score_vector
                # calculate reward vector of the current molecule based on its score vector and global Pareto Front
                reward_vector = getScoreVector(smileK, score_vector)
                # only new molecules are printed
                if affinity != 500:
                    logger.success("{}              {}".format(smileK, [round(i, 5) for i in score_vector]))
            if affinity == 500:  # affinity error in the function of CaculateAffinity of docking.py
                Update(node, REWARDMIN)
            else:
                # logger.success(smileK + '       ' + str(-affinity))
                # Update(node, -affinity)
                Update(node, reward_vector)
                updateParetoFront(smileK, score_vector)  # never add false molecule into Pareto Front
                thisScore.append(-affinity)
                thisReward.append(reward_vector)
                thisValidSmiles.append(smileK)
        else:
            # logger.error('invalid: %s' % (''.join(path)))
            Update(node, REWARDMIN)
    else:
        # logger.warning('abnormal ending: %s' % (''.join(path)))
        Update(node, REWARDMIN)

    return thisScore, thisValidSmiles, thisSmiles


def MCTS(rootNode):
    allScore = []
    allValidSmiles = []
    allSmiles = []

    currSimulationTimes = 0
    while currSimulationTimes < simulation_times:

        currSimulationTimes += 1

        # MCTS SELECT
        node, _ = Select(rootNode)

        # VisualizeInterMCTS(rootNode, modelName, './', times, QMAX, QMIN, QE)

        # rollout
        score, validSmiles, smiles = rollout(node, model)
        allScore.extend(score)
        allValidSmiles.extend(validSmiles)
        allSmiles.extend(smiles)

        # MCTS EXPAND
        atomList, logpListExpanded = sample(model, node.path, vocabulary, proVoc, smiMaxLen, proMaxLen, device, 30, protein_seq)
        pListExpanded = [np.exp(p) for p in logpListExpanded]
        Expand(node, atomList, pListExpanded)

    if args.max:  # choose the child node with maximum visits
        indices = np.argmax([n.visits for n in rootNode.childNodes])
    else:  # randomly choose by frequency
        allvisit = np.sum([n.visits for n in rootNode.childNodes]) * 1.0
        prList = np.random.multinomial(1, [n.visits / allvisit for n in rootNode.childNodes], 1)
        indices = list(set(np.argmax(prList, axis=1)))[0]
        # logger.info([n.visits / allvisit for n in rootNode.childNodes])

    return rootNode.childNodes[indices], allScore, allValidSmiles, allSmiles


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-k', type=int, default=0, help='protein index')
    parser.add_argument('--device', type=str, default='0')
    parser.add_argument('-st', type=int, default=150, help='simulation times')
    parser.add_argument('--source', type=str, default='new')
    parser.add_argument('-p', type=str, default='LT', help='pretrained model')
    parser.add_argument('--docking', type=int, default=1, help='enable docking')
    parser.add_argument('-t', type=int, default=0, help='test index')
    parser.add_argument('-g', type=int, default=0, help='gpu index')

    parser.add_argument('--max', action="store_true", help='max mode')

    args = parser.parse_args()

    if args.source == 'new':
        test_pdblist = sorted(os.listdir('./data/test_pdbs/'))
        pro_file = ['./data/test_pdbs/%s/%s_protein.pdb' % (pdb, pdb) for pdb in test_pdblist]
        ligand_file = ['./data/test_pdbs/%s/%s_ligand.sdf' % (pdb, pdb) for pdb in test_pdblist]
        protein_seq = ProteinParser(test_pdblist[args.k])
        pro_id = test_pdblist[args.k]
    else:
        raise NotImplementedError('Unknown source: %s' % args.source)

    simulation_times = args.st
    experimentId = os.path.join('experiment', args.p)
    ST = time.time()

    modelName = '30.pt'
    hpc_device = "gpu" if torch.cuda.is_available() else "cpu"
    mode = "max" if args.max else "freq"
    resFolder = '%s_%s_mcts_%s_%s_%s_%s_%s' % (hpc_device, mode, simulation_times, timeLable(), modelName, args.k, test_pdblist[args.k])

    resFolderPath = os.path.join(experimentId, resFolder)

    if not os.path.isdir(resFolderPath):
        os.mkdir(resFolderPath)
    logger.add(os.path.join(experimentId, resFolder, "{time}.log"))

    shutil.copyfile('./pareto_mcts.py', os.path.join(experimentId, resFolder) + '/multi_target_pareto_mcts.py')

    if len(protein_seq) > 999:
        logger.info('skipping %s' % test_pdblist[args.k])
    else:
        s = readSettings(experimentId)
        vocabulary = s.smiVoc
        proVoc = s.proVoc
        smiMaxLen = int(s.smiMaxLen)
        proMaxLen = int(s.proMaxLen)

        device_ids = [int(args.g)]  # 10卡机
        device = torch.device("cuda:{}".format(device_ids[0]) if torch.cuda.is_available() else "cpu")

        model = DrugTransformer(**s)
        model = torch.nn.DataParallel(model, device_ids=device_ids)  # specify the GPU ID
        model = model.to(device)  # 模型加载到设备0
        model.load_state_dict(torch.load(experimentId + '/model/' + modelName, map_location=device))
        model.to(device)
        model.eval()

        # start MCTS
        fscore = npscorer.readNPModel()  # prepare for NP-likeness score

        node = Node(path=['&'], smiMaxLen=smiMaxLen)

        times = 0
        allScores = []
        allValidSmiles = []
        allSmiles = []

        while not IsPathEnd(node.path, smiMaxLen):

            times += 1
            node, scores, validSmiles, smiles = MCTS(node)

            allScores.append(scores)
            allValidSmiles.append(validSmiles)
            allSmiles.append(smiles)

            # VisualizeMCTS(node.parentNode, modelName, resFolderPath, times)

    ET = time.time()
    logger.info('time: {} minutes'.format((ET-ST) // 60))

    mol_dic = {'protein_id': pro_id}

    pareto_mols = sorted(moleculeParetoFront.items(), key=lambda item: sum(item[1]), reverse=True)
    for i_pm, pm in enumerate(pareto_mols[:5]):
        print("Top Molecule in Pareto Front {} with scores {}".format(pm[0], infoma[pm[0]]))
        mol_dic['pareto_molecule_top_{}'.format(i_pm)] = (pm[0], infoma[pm[0]])

    print('Number of molecules {}'.format(len(infoma.keys())))
    pkl.dump(infoma, open('pareto_mcts_{}_{}.pkl'.format(pro_id, args.t), 'wb'))
    pkl.dump(mol_dic, open('./experiment/{}_{}.pkl'.format(pro_id, args.t), 'wb'))







