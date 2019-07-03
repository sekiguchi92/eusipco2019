#! /usr/bin/env python3
# coding:utf-8

import os
import soundfile as sf
import argparse
from pystoi.stoi import stoi
import numpy as np
import mir_eval

situations = ["BUS", "CAF", "PED", "STR"]
datalist_name_list = ["/home/sekiguch/Dropbox/program/data/chime/filename_list/experiment_dataset_et05_25_"+s for s in situations]

def calculate_SDR_SIR_SAR(reference_sources, estimated_sources, threshold=5):
    """ calculate  SDR, SIR, SAR
    Parameters
        reference_sources: np.array [ T or N x T ]
        estimated_sources: np.array [ T or N x T ]
        threshold: float
            when dimension of reference and estimated sources are different, score is evaluated for each pair.
            If the SDR is larger than threshold, evaluation ends.
    Results
        SDR, SIR, SAR, perm: float
    """
    if reference_sources.shape[-1] != estimated_sources.shape[-1]:
        reference_sources, estimated_sources = __align_length(reference_sources, estimated_sources)

    if (reference_sources.ndim == 1) and (estimated_sources.ndim == 1):
        sdr_sir_sar_perm = mir_eval.separation.bss_eval_sources(reference_sources, estimated_sources)
    elif (reference_sources.ndim == 2) and (estimated_sources.ndim == 2) and (reference_sources.shape[0] == estimated_sources.shape[0]):
        sdr_sir_sar_perm = mir_eval.separation.bss_eval_sources(reference_sources, estimated_sources)
    elif (reference_sources.ndim == 1) and (estimated_sources.ndim == 2):
        N = estimated_sources.shape[0]
        sdr_sir_sar_perm = np.array([-np.inf])
        for n in range(N):
            tmp = mir_eval.separation.bss_eval_sources(reference_sources, estimated_sources[n])
            if tmp[0] > threshold:
                sdr_sir_sar_perm = tmp
                break
            else:
                if tmp[0] > sdr_sir_sar_perm[0]:
                    sdr_sir_sar_perm = tmp
    elif (reference_sources.ndim == 2) and (estimated_sources.ndim == 1):
        N = reference_sources.shape[0]
        sdr_sir_sar_perm = -np.inf
        for n in range(N):
            tmp = mir_eval.separation.bss_eval_sources(reference_sources[n], estimated_sources)
            if tmp[0] > threshold:
                sdr_sir_sar_perm = tmp
                break
            else:
                if tmp[0] > sdr_sir_sar_perm[0]:
                    sdr_sir_sar_perm = tmp
    else:
        print("---Error--- input shape is invalid. [T] or [N x T] is expected, \nbut {} and {} are input.".format(reference_sources.shape, estimated_sources.shape))
        raise ValueError
    return sdr_sir_sar_perm


def calculate_PESQ_MOS(reference_sources, estimated_sources, permutation=None):
    """ calculate PESQ, MOS
    Parameters
        reference_sources: np.array [ T or N x T ]
        estimated_sources: np.array [ T or N x T ]
    Results
        score: list
            if  dimension of reference and estimated sources are both 2, the list shape is [ N x 2 ],
            else [ 2 ] (PESQ, MOS).
    """
    def __calculate_PESQ_MOS_for_one_source(reference_source, estimated_source):
        sf.write("tmp1.wav", reference_source, 16000)
        sf.write("tmp2.wav", estimated_source, 16000)
        flag = os.system("PESQ tmp1.wav tmp2.wav +16000 > tmp.txt")
        if flag != 0:
            flag = os.system("PESQ tmp1.wav tmp2.wav +16000 > tmp.txt")
            if flag != 0:
                print("---Error--- PESQ calculation error.")
                return [-1000, -1000, -1000, -1000]
        with open("tmp.txt", "r") as f:
            try:
                line = f.readlines()[-1]
                PESQ = float(line.split("\n")[0].split("\t")[0].split(" = ")[-1])
                MOS = float(line.split("\n")[0].split("\t")[1])
            except:
                print("---Error--- Some trouble occured. Returned -1000")
                return [-1000, -1000, -1000, -1000]
        return [PESQ, MOS]

    if reference_sources.shape[-1] != estimated_sources.shape[-1]:
        reference_sources, estimated_sources = __align_length(reference_sources, estimated_sources)

    if (reference_sources.ndim == 1) and (estimated_sources.ndim == 1):
        score = __calculate_PESQ_MOS_for_one_source(reference_sources, estimated_sources)
    else:
        if permutation == None:
            print("---Error--- Please set permuation parameter to decide correspondence.")
            raise ValueError
        else:
            if (reference_sources.ndim == 2) and (estimated_sources.ndim == 2) and (reference_sources.shape[0] == estimated_sources.shape[0]):
                score = []
                for n in range(reference_sources.shape[0]):
                    score.append(__calculate_PESQ_MOS_for_one_source(reference_sources[n], estimated_sources[permutation[n]]))
            elif (reference_sources.ndim == 1) and (estimated_sources.ndim == 2):
                    score = __calculate_PESQ_MOS_for_one_source(reference_sources, estimated_sources[permutation])
            elif (reference_sources.ndim == 2) and (estimated_sources.ndim == 1):
                    score = __calculate_PESQ_MOS_for_one_source(reference_sources[permutation], estimated_sources)
    return score


def calculate_STOI(reference_sources, estimated_sources, permutation=None):
    """ calculate STOI
    Parameters
        reference_sources: np.array [ T or N x T ]
        estimated_sources: np.array [ T or N x T ]
    Results
        STOI: float or list [ N ]
    """
    if reference_sources.shape[-1] != estimated_sources.shape[-1]:
        reference_sources, estimated_sources = __align_length(reference_sources, estimated_sources)

    if (reference_sources.ndim == 1) and (estimated_sources.ndim == 1):
        STOI = stoi(reference_sources, estimated_sources, 16000)
    else:
        if permutation == None:
            print("---Error--- Please set permuation parameter to decide correspondence.")
            raise ValueError
        else:
            if (reference_sources.ndim == 2) and (estimated_sources.ndim == 2) and (reference_sources.shape[0] == estimated_sources.shape[0]):
                STOI = []
                for n in range(reference_sources.shape[0]):
                    STOI.append(stoi(reference_sources[n], estimated_sources[permutation[n]]))
            elif (reference_sources.ndim == 1) and (estimated_sources.ndim == 2):
                    STOI = stoi(reference_sources, estimated_sources[permutation])
            elif (reference_sources.ndim == 2) and (estimated_sources.ndim == 1):
                    STOI = stoi(reference_sources[permutation], estimated_sources)
    return STOI


def __align_length(reference_sources, estimated_sources):
    """ align the length of input sources
    Parameters
        reference_sources: np.array [ T or N x T ]
        estimated_sources: np.array [ T or N x T ]
    Results
        reference_sources: np.array [ T or N x T ]
        estimated_sources: np.array [ T or N x T ]
    """    
    if (reference_sources.ndim == 1) and (estimated_sources.ndim == 1):
        min_len = min(len(reference_sources), len(estimated_sources))
        reference_sources = reference_sources[:min_len]
        estimated_sources = estimated_sources[:min_len]
    elif (reference_sources.ndim == 2) and (estimated_sources.ndim == 2) and (reference_sources.shape[0] == estimated_sources.shape[0]):
        min_len = min(reference_sources.shape[1], estimated_sources.shape[1])
        reference_sources = reference_sources[:, :min_len]
        estimated_sources = estimated_sources[:, :min_len]
    elif (reference_sources.ndim == 1) and (estimated_sources.ndim == 2):
        min_len = min(len(reference_sources), estimated_sources.shape[1])
        reference_sources = reference_sources[:min_len]
        estimated_sources = estimated_sources[:, :min_len]
    elif (reference_sources.ndim == 2) and (estimated_sources.ndim == 1):
        min_len = min(reference_sources.shape[1], len(estimated_sources))
        reference_sources = reference_sources[:, :min_len]
        estimated_sources = estimated_sources[:min_len]
    return reference_sources, estimated_sources


def calculate_separation_performance(reference_sources, estimated_sources, SDR=True, SIR=False, SAR=False, PESQ=False, STOI=False):
    """ calculate  SDR, PESQ, MOS and STOI from signals
    Parameters
        reference_sources: np.array [ T or N x T ]
        estimated_sources: np.array [ T or N x T ]
    Results
        SDR:  float (if SDR = True)
        SIR:  float (if SIR = True)
        SAR:  float (if SAR = True)
        PESQ: float (if PESQ = True)
        MOS:  float (if PESQ = True)
        STOI: float (if STOI = True)
    """
    reference_sources, estimated_sources = __align_length(reference_sources, estimated_sources)
    sdr_sir_sar_perm = calculate_SDR_SIR_SAR(reference_sources, estimated_sources)
    # print("sdr_sir_sar_perm:", sdr_sir_sar_perm)
    score = []
    for i, flag in enumerate([SDR, SIR, SAR]):
        if flag:
            score.append(sdr_sir_sar_perm[i])
    if PESQ:
        PESQ, MOS = calculate_PESQ_MOS(reference_sources, estimated_sources, permutation=sdr_sir_sar_perm[-1])
        score.append(PESQ)
    if STOI:
        score.append(calculate_STOI(reference_sources, estimated_sources, permutation=sdr_sir_sar_perm[-1]))
    return score


def calculate_SDR_PESQ_MOS_STOI(wav_org, wav_sep, threshold=np.inf):
    """ calculate  SDR, PESQ, MOS and STOI from signals
    Parameters
        wav_org: np.array [ T ]
        wav_sep: np.array [ T or N x T ]
    Results
        SDR:  float
        PESQ: float
        MOS:  float
        STOI: float
    """
    def _calcualte_SDR_PESQ_MOS_STOI(wav_org, wav_sep):
        min_len = min(len(wav_org), len(wav_sep))
        tmp1 = wav_org[:min_len]
        tmp2 = wav_sep[:min_len]
        sf.write("tmp1.wav", tmp1, 16000)
        sf.write("tmp2.wav", tmp2, 16000)
        flag = os.system("PESQ tmp1.wav tmp2.wav +16000 > tmp.txt")
        if flag != 0:
            print("Error: PESQ error")
            flag = os.system("PESQ tmp1.wav tmp2.wav +16000 > tmp.txt")
            if flag != 0:
                print("Error: PESQ error \n stop processing")
                return [-1000, -1000, -1000, -1000]

        SDR, sir, sar, perm = mir_eval.separation.bss_eval_sources(tmp1, tmp2)

        with open("tmp.txt", "r") as f:
            try:
                line = f.readlines()[-1]
                PESQ = float(line.split("\n")[0].split("\t")[0].split(" = ")[-1])
                MOS = float(line.split("\n")[0].split("\t")[1])
            except:
                print("Some trouble occured \n return -1000")
                return [-1000, -1000, -1000, -1000]

        STOI = stoi(tmp1, tmp2, 16000)

        return [SDR, PESQ, MOS, STOI]

    if wav_sep.ndim > 1:
        if (wav_sep.shape[0] > wav_sep.shape[1]):
            wav_sep = wav_sep.T
        N = wav_sep.shape[0]
        max_SDR  = -np.inf
        max_PESQ = -np.inf
        max_MOS  = -np.inf
        max_STOI = -np.inf
        max_index = 0
        for n in range(N):
            SDR, PESQ, MOS, STOI = _calcualte_SDR_PESQ_MOS_STOI(wav_org, wav_sep[n])
            if SDR > max_SDR:
                max_SDR = SDR
                max_PESQ = PESQ
                max_MOS  = MOS
                max_STOI = STOI
                max_index = n
            if max_SDR > threshold:
                break
        print("max_index : ", max_index)
    else:
        max_SDR, max_PESQ, max_MOS, max_STOI = _calcualte_SDR_PESQ_MOS_STOI(wav_org, wav_sep)

    return [max_SDR, max_PESQ, max_MOS, max_STOI]


def calculate_SDR(wav_org, wav_sep, threshold=np.inf):
    """ calculate SDR
    Parameters
        wav_org: np.array [ T ]
        wav_sep: np.array [ T or N x T ]
        threshold: int
    Results
        SDR:  float
    """
    def _calcualte_SDR(wav_org, wav_sep):
        min_len = min(len(wav_org), len(wav_sep))
        tmp1 = wav_org[:min_len]
        tmp2 = wav_sep[:min_len]

        SDR, sir, sar, perm = mir_eval.separation.bss_eval_sources(tmp1, tmp2)

        return SDR

    if wav_sep.ndim > 1:
        if (wav_sep.shape[0] > wav_sep.shape[1]):
            wav_sep = wav_sep.T
        N = wav_sep.shape[0]
        max_SDR  = -np.inf
        max_index = 0
        for n in range(N):
            SDR = _calcualte_SDR(wav_org, wav_sep[n])
            if SDR > max_SDR:
                max_SDR = SDR
                max_index = n
            if max_SDR > threshold:
                break
        print("max_index 1: ", max_index)
    elif wav_org.ndim > 1:
        N = wav_org.shape[0]
        max_SDR  = -np.inf
        max_index = 0
        for n in range(N):
            SDR = _calcualte_SDR(wav_org[n], wav_sep)
            if SDR > max_SDR:
                max_SDR = SDR
                max_index = n
            if max_SDR > threshold:
                break
        print("max_index 2: ", max_index)
    else:
        max_SDR = _calcualte_SDR(wav_org, wav_sep)

    return max_SDR


# def calculate_SDR_SAR_SIR(wav_org, wav_sep):
    # """ calculate SDR
    # Parameters
    #     wav_org : N x T1
    #     wav_sep : N x T2
    # Results
    #     SDR:  float
    # """

    # if wav_sep.ndim > 1:
    #     min_length = min(wav_org.shape[1], wav_sep.shape[1])
    #     SDR, SIR, SAR, perm = mir_eval.separation.bss_eval_sources(wav_org[:, :min_length], wav_sep[:, :min_length])
    #     print(perm)
    #     return SDR, SIR, SAR
    # else:
    #     print("wav.shape should be N x T")
    #     raise ValueError



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(       '--NUM_noise', type=  int, default=       1, help='number of noise')
    parser.add_argument('--NUM_basis_speech', type=  int, default=      10, help='number of basis of speech (mode_speech=NMF)')
    parser.add_argument( '--NUM_basis_noise', type=  int, default=       4, help='number of basis of noise (mode_noise=NMF)')
    parser.add_argument(     '--mode_speech', type=  str, default=   "VAE", help='VAE or NMF')
    parser.add_argument(      '--mode_noise', type=  str, default=   "NMF", help='VAE or NMF')
    parser.add_argument( "--mode_covariance", type=  str, default="FullRank", help="FullRank or Rank1")
    parser.add_argument(   '--mode_update_Z', type=  str, default="backprop", help='sampling, sampling2, backprop, backprop2, hybrid, hybrid2')
    parser.add_argument('--mode_initialize_covarianceMatrix', type=str, default="obs", help='cGMM, cGMM2, unit, obs')
    args = parser.parse_args()

    if args.mode_speech == "VAE":
        if (args.mode_covariance == "FullRank"):
            method_type = "FullRank"
        else:
            method_type = "Rank1"
    elif args.mode_speech == "NMF":
        if (args.mode_covariance == "FullRank"):
            method_type = "MNMF"
        else:
            method_type = "Rank1"

    args.NUM_basis_speech = args.NUM_basis_noise

    if method_type == "FullRank":
        filename_prefix_wo_ID = "{}-sep-Wiener-N={}-it=100-itZ=50-speech=VAE-Ls=NONE-noise=NMF-Ln={}-D=16-scale=none-init={}-latent={}-DNN=simon_e1_d1_IS".format(method_type, args.NUM_noise, args.NUM_basis_noise, args.mode_initialize_covarianceMatrix, args.mode_update_Z)
        dir_separated = "/n/sd2/sekiguchi/data_for_paper/data-taslp2018/FullRank/"
    elif (method_type == "Rank1") and (args.mode_speech == "VAE"):
        filename_prefix_wo_ID = "{}-sep-Linear-it=100-itZ=50-speech=VAE-Ls=NONE-noise=NMF-Ln={}-D=16-scale=none-init={}-latent={}-DNN=simon_e1_d1_IS".format(method_type, args.NUM_basis_noise, args.mode_initialize_covarianceMatrix, args.mode_update_Z)
        dir_separated = "/n/sd2/sekiguchi/data_for_paper/data-taslp2018/Rank1/"
    elif method_type == "MNMF":
        filename_prefix_wo_ID = "{}-sep-Wiener-N={}-it=100-L={}-init={}".format(method_type, args.NUM_noise+1, args.NUM_basis_noise, args.mode_initialize_covarianceMatrix)
        dir_separated = "/n/sd2/sekiguchi/data_for_paper/data-taslp2018/FullRank/"
    elif (method_type == "Rank1") and (args.mode_speech == "NMF"):
        filename_prefix_wo_ID = "{}-sep-Linear-it=100-Ls={}-Ln={}-init={}".format(method_type, args.NUM_basis_speech, args.NUM_basis_noise, args.mode_initialize_covarianceMatrix)
        dir_separated = "/n/sd2/sekiguchi/data_for_paper/data-taslp2018/ILRMA/"

    dir_original = "/n/sd2/sekiguchi/CHiME3/mydata/et05_simu_clean/"
    # dir_original = "/n/sd2/sekiguchi/CHiME3/data/et05/simu_orig_clean/"

    if args.mode_covariance == "FullRank":
        N = args.NUM_noise + 1
    else:
        N = 5

    for datalist_name in datalist_name_list:
        with open(datalist_name, "r") as f:
            for line in f.readlines():
                file_id = line.split(".")[0].split("/")[-1]
                filename_org = dir_original + file_id + ".CH5.wav"
                # filename_org = dir_original + file_id.split("_")[1].lower() + ".wav"
                filename_sep = dir_separated + filename_prefix_wo_ID + "-ID={}.wav".format(file_id)

                wav_org = sf.read(filename_org)[0]
                wav_sep = sf.read(filename_sep)[0].T

                print(calculate_PESQ(wav_org, wav_sep))
