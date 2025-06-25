import argparse
import os
import pickle
import re

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

    match = re.match(r"(.*)Corr[Z|W]\.pkl\.lz4", os.path.basename(corrf))
    return match[1]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c",
        "--corr",
        type=str,
        help="Correction to inspect",
        required=True,
    )
    parser.add_argument(
        "-o",
        "--outfolder",
        type=str,
        help="Full foldername to store plots in",
        required=True,
    )

    parser.add_argument("--debug", action="store_true", help="Print debug output")
    parser.add_argument(
        "--noColorLogger",
        action="store_true",
        default=False,
        help="Do not use logging with colors",
    )
    # parser.add_argument(
    #     "--outpath",
    #     type=str,
    #     default=f"{common.data_dir}/TheoryCorrections",
    #     help="Output path",
    # )

    return parser.parse_args()


# Define the exponential model: A * exp(-ax)
def exp_func(x, A, a):
    return A * np.exp(a*x)


def main():
    args = parse_args()
    logger = logging.setup_logger(
        "inspect_theory_corr", 4 if args.debug else 3, args.noColorLogger
    )
    os.makedirs(args.outfolder, exist_ok=True)

    corrfile = pickle.load(lz4.frame.open(args.corr))
    proc = "Z" if "CorrZ" in args.corr else "W"
    corrname=corr_name(args.corr)
    var = "qT"

    print(corrfile[proc][corrname + "_minnlo_ratio"][{"vars" : 0}])
    ratio_to_minnlo = corrfile[proc][corrname + "_minnlo_ratio"][{"vars" : 0}].project(var)
    nnlojet_fo = input_tools.read_nnlojet_pty_hist("/work/submit/areimers/wmass/TheoryCorrections/NNLOjet/Z/ZjNNLO/final/ptz")
    nnlojet_fo_smooth_pt = hh.smooth_hist(nnlojet_fo.project("qT", "vars"), "qT", start_bin=4)
    scetlib_n4ll = input_tools.read_scetlib_hist("/work/submit/areimers/wmass/TheoryCorrections/SCETlib/ct18z_nplambda_n4+0ll/inclusive_Z_CT18Z_nplambda_N4+0LL_combined.pkl")[{"vars" : 0}].project(var)
    scetlib_n3lo_sing = (-1*input_tools.read_scetlib_hist("/work/submit/areimers/wmass/TheoryCorrections/SCETlib/ct18z_nplambda_n4+0ll/inclusive_Z_CT18Z_nplambda_n3lo_sing.pkl")[{"vars" : 0}]).project(var)

    nnlojet_fo = nnlojet_fo[{"vars" : 0}].project(var)
    nnlojet_fo_smooth_pt = nnlojet_fo_smooth_pt[{"vars" : 0}].project(var)
    
    centers_nnlojet_fo = 0.5 * (nnlojet_fo.axes[0].edges[:-1] + nnlojet_fo.axes[0].edges[1:])
    values_nnlojet_fo = nnlojet_fo.values()
    errors_nnlojet_fo = np.sqrt(nnlojet_fo.variances())
    fit_mask = (centers_nnlojet_fo > 10) & (centers_nnlojet_fo < 60)  

    popt, pcov = curve_fit(exp_func, centers_nnlojet_fo[fit_mask], values_nnlojet_fo[fit_mask], sigma=errors_nnlojet_fo[fit_mask], absolute_sigma=True)
    A_fit, lambda_fit = popt


    plt.figure()
    # ratio_to_minnlo.plot()
    mplhep.histplot(ratio_to_minnlo, yerr=False)
    plt.xlabel("qT [GeV]")
    plt.ylabel("Ratio to MiNNLO")
    plt.grid(True)
    plt.savefig(f"{args.outfolder}/{corrname}_to_MiNNLO_{var}.pdf")
    plt.close()

    corr_hist = corrfile[proc][corrname + "_hist"][{"vars" : 0}].project(var)

    # print(nnlojet_fo)
    plt.figure()
    # mplhep.histplot(corr_hist, label="N3LO + N4LL", yerr=True)
    mplhep.histplot(nnlojet_fo, label="N3LO FO", yerr=True)
    # mplhep.histplot(nnlojet_fo_smooth_pt, label="N3LO FO (smoothed)", yerr=True)
    # mplhep.histplot(scetlib_n4ll, label="N4LL resumm", yerr=True)
    # mplhep.histplot(scetlib_n3lo_sing, label="N3LO sing", yerr=True)
    xfit = np.linspace(centers_nnlojet_fo[0], centers_nnlojet_fo[-1], 500)
    plt.plot(xfit, exp_func(xfit, *popt), label=f"Fit: A exp(-x / {lambda_fit:.1f})", color="red")
    plt.xlabel("qT [GeV]")
    plt.ylabel(f"{corrname}")
    plt.yscale("log")
    plt.grid(True)
    plt.legend()
    plt.savefig(f"{args.outfolder}/{corrname}_hist_{var}.pdf")
    plt.close()

    minnlo_hist = corrfile[proc]["minnlo_ref_hist"]
    plt.figure()
    minnlo_hist.project(var).plot()
    plt.xlabel("qT [GeV]")
    plt.ylabel(f"MiNNLO")
    plt.grid(True)
    plt.savefig(f"{args.outfolder}/MiNNLO_hist_{var}.pdf")
    plt.close()

    


    fig = plot_tools.makePlotWithRatioToRef(
        hists=[
            nnlojet_fo_smooth_pt,
            nnlojet_fo,
        ],
        labels=[
             "N3LO FO Smoothed pT only",
             "N3LO FO Unsmoothed",
            ],
        colors=[
             "black",
             "purple",
            ],
        xlabel="q$_{T}$ (GeV)", 
        ylabel="Events/bin",
        rlabel="x/smoothed",
        rrange=[0.9, 1.1],
        nlegcols=1,
        xlim=None, binwnorm=1.0, baseline=True, 
        ratio_legend=False,
        yerr=True,
        yerr_ratio=True,
        linewidth=1,
    )
    fig.savefig(f"{args.outfolder}/N3LO_FO_{var}_smothings.pdf")
    
    

if __name__ == "__main__":
    main()
