import os
# from reportlab.lib.units import cm
# import Bio
# from Bio import SeqIO
# from Bio.Graphics import BasicChromosome
import pandas as pd
from helperMethods import *
from conf import Conf
import numpy as np
import random
import csv
import pickle

# from colour import Color
# from reportlab.lib import colors
# import numpy as np
# from reportlab.lib import colors
# import matplotlib.pyplot as plt
# import matplotlib.image as mpimg

methylFolderPath = 'C:/Users/alonal/Google Drive/phd/methylation_dropbox/methylation'
# entries = ["../res/GCF_000001405.37_GRCh38.p11_genomic.gbff"]
entries = ["C:/Users/agotl/Downloads/GCF_000001405.37/ncbi_dataset/data/GCF_000001405.37/genomic.gbff"]
"C:/Users/agotl/PycharmProjects/Methylation/res/for_distances/GCF_000001405.37 (2).zip/ncbi_dataset/data/GCF_000001405.37/genomic.gbff"


# methylFolderPath = "../"

def _getGenesInData(expression=Conf.dfExpression):
    path = methylFolderPath + "/res/"
    genes = pd.read_csv(expression)["rownames"].unique()
    # genes2 = pd.read_csv(path+'TCGA_CH3_final.csv', nrows=1)
    # print genes2
    # print genes
    return genes


def createProbePositionsDict(filename=Conf.dfMethyl):
    probesInData = np.array(pd.read_csv(filename, usecols=[0]))
    probesInData.sort()
    probePosData = pd.read_csv('../res/450k_ESR1_FOXA1_GATA3.txt', delimiter="\t")
    probeToPos = {}
    for i in range(len(probePosData)):
        if probePosData.loc[i, 'Probe'] in probesInData:
            probeToPos[probePosData.loc[i, 'Probe']] = [probePosData.iloc[i, 0].replace('chr', ''),
                                                        probePosData.iloc[i, 1], probePosData.iloc[i, 2]]
        print(i)
        # if i >10:
        #     break
    save_obj(probeToPos, 'probeToPos', '../res/')


def createDistanceMatrx(sort_probes, ch3_filename=Conf.dfMethyl, e_filename=Conf.dfExpression, numProbes=-1,
                        preSelectedProbes=False, genesInCols=False, useInverseDist=False, window_limit=-1):
    """
    Creates distances.csv (between gene expression and CpG.
    :param ch3_filename:
    :param e_filename:
    :param numProbes:
    :param preSelectedProbes:
    :return:
    """

    probesInData = np.array(pd.read_csv(ch3_filename, usecols=[0])).flatten()
    if genesInCols:
        genesInData = np.array(pd.read_csv(e_filename, nrows=0).columns)
        genesInData = genesInData[1:]
    else:
        genesInData = pd.read_csv(e_filename, usecols=[0])
        genesInData = np.array(genesInData)[:, 0]
    cols = ['Probe']
    # data = []
    genesKept = []
    probesKept = []
    dataToAppend = False
    probeCounter = 0
    first_row_being_added = True

    try:
        os.remove('../res/distances.csv')
    except:
        print("distances file not removed because wasn't found")
    if numProbes != -1:
        chosenProbes = random.sample(list(probesInData), numProbes)
    else:
        chosenProbes = probesInData
    if sort_probes:
        chosenProbes.sort()  # sort only if later we want that in dataProcessor we will only have to sort the CH3 file and not the distances file
    with open('../res/distances.csv', 'a') as dist_file:
        print(chosenProbes)
        for p in chosenProbes:
            # if not preSelectedProbes and numProbes != -1:
            if numProbes != -1:
                if probeCounter >= numProbes:
                    break
            # if probeCounter % 1000 == 0:
            #     print(probeCounter)
            # if probeCounter >= 100:
            #     break
            probe = p
            try:
                probeTuple = probeToPos[probe]
                probesKept.append(probe)
                dataRow = [probe]
                row_has_pos_dist_in_window = False

            except:
                continue
            for gene in genesInData:
                # gene = g[0]
                try:
                    geneTuple = geneToPos[gene]
                    if int(geneTuple[1]) > int(geneTuple[2]):
                        print("FOUND ONE!")
                    probeChr = str(probeTuple[0]).upper()
                    geneChr = str(geneTuple[0]).upper()
                    if probeChr == geneChr:
                        if useInverseDist:
                            # import math
                            distance = 1 / float(int(geneTuple[1]) - int(probeTuple[1]))
                            if window_limit != -1:
                                if abs(1 / float(distance)) <= window_limit:
                                    row_has_pos_dist_in_window = True
                        else:
                            distance = int(geneTuple[1]) - int(probeTuple[1])

                        dataRow.append(distance)
                    else:
                        if useInverseDist:
                            dataRow.append(0)  # the length of chr1 which is the largest possible distance
                        else:
                            dataRow.append(248956422)  # the length of chr1 which is the largest possible distance
                    if probeCounter == 0:  # only on the first iteration over all genes add them as column names
                        cols.append(gene)
                        genesKept.append(gene)
                except:
                    continue

            if useInverseDist and window_limit != -1:
                if row_has_pos_dist_in_window:
                    if first_row_being_added:
                        dist_file.write(','.join(cols))
                        first_row_being_added = False
                    dist_file.write('\n')
                    dist_file.write(','.join(map(str, dataRow)))
                probeCounter += 1
            else:
                if probeCounter == 0:
                    dist_file.write(','.join(cols))
                dist_file.write('\n')
                dist_file.write(','.join(map(str, dataRow)))
                probeCounter += 1

    save_obj(genesKept, "genesKept", '../res/')
    save_obj(probesKept, "probesKept", '../res/')


def createDistanceMatrx_adjusted(geneToPos, probeToPos, sort_probes, numProbes=-1, preSelectedProbes=False,
                                 genesInCols=False, useInverseDist=False, window_limit=-1,
                                 distance_path="distances.csv"):
    """
    Creates distances.csv (between gene expression and CpG.
    """
    import math
    probesInData = probeToPos['probe'].unique()
    genesInData = geneToPos['gene'].unique()
    cols = ['Probe']
    # data = []
    genesKept = []
    probesKept = []
    dataToAppend = False
    probeCounter = 0
    first_row_being_added = True
    distaces_df = pd.DataFrame(columns = ["Probe"] + list(genesInData))
    distaces_df.to_csv('Distances2k.csv')

    cnt = 0
    # print(f"distaces_df : {distaces_df}")

    try:
        os.remove(distance_path)
        print("distances file was removed")
    except:
        print("distances file not removed because wasn't found")
    if numProbes != -1:
        chosenProbes = random.sample(list(probesInData), numProbes)
    else:
        if preSelectedProbes != False:
            chosenProbes = preSelectedProbes
            print(f"the chosen probes are preselected: \n {chosenProbes}")
        else:
            chosenProbes = probesInData
    if sort_probes:
        chosenProbes.sort()  # sort only if later we want that in dataProcessor we will only have to sort the CH3 file and not the distances file
    with open(distance_path, 'a') as dist_file:
        print(chosenProbes)
        for probe in chosenProbes:
            if numProbes != -1 and probeCounter >= numProbes:
                break
            try:
                probesKept.append(probe)
                dataRow = [probe]
                row_has_pos_dist_in_window = False
            except:
                print(f"invalid probe {probe}")
                continue

            idx = probeToPos["probe"] == probe
            probeChr = probeToPos[idx]['chr'].iloc[0]
            cpg_start, cpg_end = probeToPos[idx]["start"], \
                                 probeToPos[idx]["end"]

            for gene in genesInData:
                try:
                    g_idx = geneToPos["gene"] == gene
                    geneChr = geneToPos[g_idx]['chromosome'].iloc[0]
                    if str(probeChr) == str(geneChr):
                        gene_start, gene_end = geneToPos[g_idx]["start"], geneToPos[g_idx]["end"]
                        if useInverseDist:
                            distance = 1 / float(int(gene_start) - int(cpg_start))
                            if window_limit != -1:
                                if abs(1 / float(distance)) <= window_limit:
                                    row_has_pos_dist_in_window = True
                        else:
                            distance = int(gene_start) - int(cpg_start)
                        dataRow.append(distance)
                    else:
                        if useInverseDist:
                            dataRow.append(0)  # the length of chr1 which is the largest possible distance
                        else:
                            dataRow.append(248956422)  # the length of chr1 which is the largest possible distance
                    if probeCounter == 0:  # only on the first iteration over all genes add them as column names
                        cols.append(gene)
                        genesKept.append(gene)
                except:
                    continue

            if useInverseDist and window_limit != -1:
                if row_has_pos_dist_in_window:
                    if first_row_being_added:
                        dist_file.write(','.join(cols))
                        first_row_being_added = False

                    if cnt % 1 == 0:
                        print(f"Cpg idx : {idx.index[idx]}")
                    cnt = cnt + 1

                    dist_file.write('\n')
                    dist_file.write(','.join(map(str, dataRow)))
                probeCounter += 1
            else:
                if probeCounter == 0:
                    dist_file.write(','.join(cols))

                dist_file.write('\n')
                dist_file.write(','.join(map(str, dataRow)))
                probeCounter += 1

            # if useInverseDist : #and window_limit != -1:
            #     if row_has_pos_dist_in_window:
            #         cnt = cnt + 1
            #         if first_row_being_added:
            #             print("first_row_being_added")
            #             # dist_file.write(','.join(cols))
            #             first_row_being_added = False
            #
            #         df = pd.DataFrame([dataRow], columns=["Probe"] + list(genesInData))
            #         distaces_df = distaces_df.append(df, ignore_index=True)
            #
            #         # if(cnt % 10 == 0):
            #         distaces_df.to_csv('Distances2k.csv', mode='a', index=False, header=False)
            #             # distaces_df.drop(distaces_df.index, inplace=True)

                # probeCounter += 1
            # else:
                # if probeCounter == 0:
                #     dist_file.write(','.join(cols))
                # dist_file.write('\n')
                # dist_file.write(','.join(map(str, dataRow)))

                # probeCounter += 1

    # distaces_df.to_csv('Distances2k.csv', mode='a', index=False, header=False)
    #distaces_df.to_csv('Distances2k.csv')
    print(f"finished distances, probeCounter is {probeCounter}, and saved in {distance_path}")


# TODO: DEPRECATED!!!!!!!!!!!!!!!!!!!!
def createDistancesFileGivenChosenData(probes, genes, probeToPos, geneToPos, suffix):
    probeCounter = 0
    cols = ['Probe']
    with open('../res/distances' + suffix + '.csv', 'a') as the_file:
        for probe in probes:
            if probeCounter % 1000 == 0:
                print(probeCounter)
            # probe = p[0]
            dataRow = [probe]
            probeTuple = probeToPos[probe]
            for gene in genes:
                # gene = g[0]
                try:
                    geneTuple = geneToPos[gene]
                    probeChr = str(probeTuple[0]).upper()
                    geneChr = str(geneTuple[0]).upper()
                    if probeChr == geneChr:
                        distance = int(geneTuple[1]) - int(probeTuple[1])
                        dataRow.append(distance)
                    else:
                        dataRow.append(0)
                    if probeCounter == 0:
                        cols.append(gene)
                except:
                    pass
            # data.append(dataRow)
            if probeCounter == 0:
                the_file.write(','.join(cols))
            the_file.write('\n')
            the_file.write(','.join(map(str, dataRow)))
            probeCounter += 1


def createGenePositionsDict(expression=Conf.dfExpression, gene_to_pos_csv_path=""):
    genesInData = _getGenesInData(expression=expression)
    geneToPos = {}
    from Bio import SeqIO

    # get the gene's pos and insert into dict.
    counter = 0
    for index, filename in enumerate(entries):
        records = SeqIO.parse(filename, "genbank")
        for record in records:
            for fg in record.features:
                if "gene" in fg.qualifiers:
                    genes = fg.qualifiers['gene']
                    for gene in genes:
                        if str(gene).upper() in genesInData:
                            start = fg.location.start.position
                            end = fg.location.end.position
                            for f in record.features:
                                try:
                                    chromosome = f.qualifiers['chromosome'][0]
                                except:
                                    continue
                            geneToPos[gene] = [chromosome, start, end]
                            counter += 1
                            print("\n" + gene + " " + str(chromosome) + "_" + str(start) + "_" + str(end))

    with open(gene_to_pos_csv_path, "w") as outfile:
        writer = csv.writer(outfile)
        genes_list = list(geneToPos.keys())
        columns = ['gene', 'chromosome', 'start', 'end']
        writer.writerow(columns)
        for i, gene in enumerate(genes_list):
            row = [gene] + list(geneToPos[gene])
            writer.writerow(row)

    return geneToPos


def convertPosDicsToCsv():
    probeToPos = load_obj('probeToPos', '../res/')
    geneToPos = load_obj('geneToPos', '../res/')

    w = csv.writer(open("probeToPos.csv", "w"))
    for key, val in probeToPos.items():
        w.writerow([key, val])

    w = csv.writer(open("geneToPos.csv", "w"))
    for key, val in geneToPos.items():
        w.writerow([key, val])


# TODO: Deprecated
# def getDistToRandomProbesNotInKeptData(numProbes, run_example, using_kept_probes=False):
#     '''
#     Used for cases where we want to add CpGs as input and then we want to add CpG-CpG distances
#     :param numProbes:
#     :param run_example:
#     :return:
#     '''
#     probeToPos = load_obj("probeToPos", '../res/')
#     if run_example:
#         probesKept = load_obj("probesKept_sample", '../res/')
#     else:
#         probesKept = load_obj("probesKept", '../res/')
#     probesNotKept = set(probeToPos) - set(probesKept)
#     newProbes = random.run_example(probesNotKept, numProbes)
#     out = []
#     counter = 0
#     for p in probesKept:
#         if counter % 1000 == 0:
#             print(counter)
#         row = []
#         row.append(p)
#         if run_example:
#             try:
#                 pTuple = probeToPos[p]
#                 pChr = str(pTuple[0]).upper()
#                 for n in newProbes:
#                     nTuple = probeToPos[n]
#                     nChr = str(nTuple[0]).upper()
#                     if pChr == nChr:
#                         distance = int(nTuple[1]) - int(pTuple[1])
#                         row.append(distance)
#                     else:
#                         row.append(0)
#             except:
#                 row.extend([0 for i in newProbes])
#         else:
#             pTuple = probeToPos[p]
#             pChr = str(pTuple[0]).upper()
#             for n in newProbes:
#                 nTuple = probeToPos[n]
#                 nChr = str(nTuple[0]).upper()
#                 if pChr == nChr:
#                     distance = int(nTuple[1]) - int(pTuple[1])
#                     row.append(distance)
#                 else:
#                     row.append(0)
#
#         out.append(row)
#         counter += 1
#     outDataFrame = pd.DataFrame(out)
#     cols = ['Probe']
#     cols.extend(newProbes)
#     outDataFrame.columns = cols
#     outDataFrame.to_csv('../res/'+'distancesCpGToCpG.csv', index=None)
#     # as run_example run this with numProbes = 3 and probesKept_sample


def createProbePositionsDict_adjusted(methyl_path, prob_to_pos_csv_path):
    prob_to_pos = pd.read_csv(methyl_path, header=0, sep='\t', dtype={'Chr': object})
    prob_to_pos = prob_to_pos.drop(columns=['chromHMM'])
    prob_to_pos['chr'] = [c[3:] for c in prob_to_pos['chr']]
    prob_to_pos = prob_to_pos.reindex(['probe', 'start', 'end', 'chr'], axis=1)
    prob_to_pos.to_csv(prob_to_pos_csv_path)
    return prob_to_pos


if __name__ == '__main__':
    # getDistToRandomProbesNotInKeptData(4, run_example=True)
    # gene_to_pos = createGenePositionsDict(expression = "C:/Users/agotl/Downloads/Normal_expression_with_index.csv", gene_to_pos_csv_path = "C:/Users/agotl/PycharmProjects/Methylation/res/for_distances/geneToPos.csv")
    # prob_to_pos = createProbePositionsDict_adjusted(methyl_path="C:/Users/agotl/PycharmProjects/Methylation/res/res_full_train_data/450k_probes_ChroMM.bed", prob_to_pos_csv_path = "C:/Users/agotl/PycharmProjects/Methylation/res/for_distances/prob_to_pos.csv")
    #

    gene_to_pos = pd.read_csv("../res/geneToPos.csv")
    prob_to_pos = pd.read_csv("../res/prob_to_pos.csv")

    # creating distance file sampled (with the sampled CPGs used in the labels file)
    distance_path = "../res/distances_2k.csv"
    createDistanceMatrx_adjusted(geneToPos=gene_to_pos, probeToPos=prob_to_pos,
                                 numProbes=-1, sort_probes=True, preSelectedProbes=False, useInverseDist=True,
                                 window_limit=2000, distance_path=distance_path)

    # run_example:
    # createMatrixOfPosDist(ch3_filename='../res/combined_CH3_sample_nlargest.csv', e_filename='../res/combined_E.csv', preSelectedProbes=True)

    # g = getGenesInData()
    # print g
    # print len(g)
    # o = load_obj("probeToPos", methylFolderPath+'res/')
    # print len(o)
    # print o[o.keys()[0]]
    # c = 0
    # for i in g:
    #     try:
    #         o[i]
    #     except:
    #         c+=1
    #         print i
    # print c