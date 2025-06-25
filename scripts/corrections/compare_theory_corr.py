import argparse
import os
import pickle
import re
import math

import lz4.frame

from utilities import common
from wremnants import theory_corrections
from wums import boostHistHelpers as hh
from wums import logging, output_tools, plot_tools
from scripts.corrections.make_theory_corr import read_corr
import matplotlib.pyplot as plt
from utilities.io_tools import input_tools
import mplhep
import numpy as np
from scipy.optimize import curve_fit


def corr_name(corrf):
    if not corrf.endswith(".pkl.lz4"):
        raise ValueError(f"File {corrf} is not a lz4 compressed pickle file")

    match = re.match(r"(.*)Corr[Z|W]\.pkl\.lz4", corrf)
    return match[1]



def main():
    
    corrections_folder = "/work/submit/areimers/wmass/WRemnants/wremnants-data/data/TheoryCorrections"
    corrections = ["scetlib_nnlojetN4p0LLN3LOUnsmoothedCorrZ.pkl.lz4", "scetlib_nnlojet_N3p0LLN2LOUnsmoothedCorrZ.pkl.lz4", "scetlib_dyturboCorrZ.pkl.lz4"]

    # os.makedirs(args.outfolder, exist_ok=True)

    corrfiles = [pickle.load(lz4.frame.open(os.path.join(corrections_folder, c))) for c in corrections]
    proc = "Z"
    corrnames = [corr_name(c) for c in corrections]
    var = "qT"

    nnlojet_fo = input_tools.read_nnlojet_pty_hist("/work/submit/areimers/wmass/TheoryCorrections/NNLOjet/Z/ZjNNLO/final/ptz")

    for cname, cfile in zip(corrnames, corrfiles):
        print(f"\n\n--> Now at correction {cname}")
        # print(cfile[proc])
        h = cfile[proc][cname+"_hist"][{"vars": 0}]
        relstatunc = math.sqrt(h.sum().variance) / h.sum().value
        print(f"Relative stat unc of correction: {relstatunc}")
        h_minnlo = cfile[proc]["minnlo_ref_hist"]
        relstatunc_minnlo = math.sqrt(h_minnlo.sum().variance) / h_minnlo.sum().value
        print(f"Relative stat unc of MiNNLO reference: {relstatunc_minnlo}")
        sf = relstatunc / relstatunc_minnlo
        print(f"Should increase the MiNNLO stat unc by a factor of {sf:.02f} to reflect the stat unc in the correction")
        

        # ratio_to_minnlo = cfile[proc][cname + "_minnlo_ratio"][{"vars" : 0}].project(var)
        # nnlojet_fo_smooth_pt = hh.smooth_hist(nnlojet_fo.project("qT", "vars"), "qT", start_bin=4)
        # scetlib_n4ll = input_tools.read_scetlib_hist("/work/submit/areimers/wmass/TheoryCorrections/SCETlib/ct18z_nplambda_n4+0ll/inclusive_Z_CT18Z_nplambda_N4+0LL_combined.pkl")[{"vars" : 0}].project(var)
        # scetlib_n3lo_sing = (-1*input_tools.read_scetlib_hist("/work/submit/areimers/wmass/TheoryCorrections/SCETlib/ct18z_nplambda_n4+0ll/inclusive_Z_CT18Z_nplambda_n3lo_sing.pkl")[{"vars" : 0}]).project(var)

        

if __name__ == "__main__":
    main()
