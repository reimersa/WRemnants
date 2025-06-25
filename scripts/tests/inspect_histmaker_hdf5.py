import os
import pickle

import lz4.frame

from wums import boostHistHelpers as hh
from wums import output_tools, plot_tools
import matplotlib.pyplot as plt
from utilities.io_tools import input_tools
import mplhep
import numpy as np
from scipy.optimize import curve_fit




def main():
    plot_variations = True

    corrname = "scetlib_nnlojet_N3p0LLN2LOUnsmoothedCorr"
    # corrname = "scetlib_nnlojet_N4p0LLN3LOUnsmoothed_N3pLLFixedCorr"
    # corrname = "scetlib_nnlojetN4p0LLN3LOUnsmoothedCorr"

    # infilename = f"/scratch/submit/cms/areimers/wmass/histmaker/mz_dilepton_{corrname}_maxFiles_m1.hdf5"
    # infilename = f"//scratch/submit/cms/areimers/wmass/histmaker/mz_dilepton_{corrname}_maxFiles_1__perBinStatCorrUnc.hdf5"
    infilename = f"//scratch/submit/cms/areimers/wmass/histmaker/mz_dilepton_{corrname}_maxFiles_1_perBinStatCorrUncNdim.hdf5"

    # outfolder = f"/work/submit/areimers/wmass/plots/histmaker/Z/variations/{corrname}"
    # outfolder = f"/work/submit/areimers/wmass/plots/histmaker/Z/variations/{corrname}_perBinStatCorrUnc"
    outfolder = f"/work/submit/areimers/wmass/plots/histmaker/Z/variations/{corrname}_perBinStatCorrUncNdim"

    samplename = "ZmumuPostVFP"
    histname = f"nominal_ptll_{corrname}"
    os.makedirs(outfolder, exist_ok=True)
    hist = input_tools.read_and_scale(infilename, samplename, histname)

    # print(hist)

    # compare statistical uncertainties
    # print(hist[{"vars": 0}])
    # relstatunc_nnlojet = hist[{"vars": 0}].sum().variance / hist[{"vars": 0}].sum().value
    # print(relstatunc_nnlojet)
    # sqrt(h.sum().variance)/h.sum().value


    if plot_variations:
        vars = list(hist.axes["vars"])
        nominal = hist[{"vars" : 0}]
        print(len(vars), vars)
        raise ValueError("stop")
        for var in vars:
            varhist = hist[{"vars" : var}]

            fig = plot_tools.makePlotWithRatioToRef(
                hists=[
                    nominal,
                    varhist,
                ],
                labels=[
                     "Nominal",
                     f"{var}",
                    ],
                colors=[
                     "black",
                     "red",
                    ],
                xlabel="p$_{T}^{\ell\ell}$ (GeV)", 
                ylabel="Events/bin",
                rlabel="x/nominal",
                rrange=[0.9, 1.1],
                nlegcols=1,
                xlim=None, binwnorm=1.0, baseline=True, 
                ratio_legend=False,
                yerr=False,
                yerr_ratio=False,
                linewidth=1,
            )
            fig.savefig(f"{outfolder}/{var}.pdf")
            plt.close(fig)

        colors = ["red", "blue", "green", "orange", "pink"]
        for idxplot in range(0, len(vars), 5):
            varnames = vars[idxplot:idxplot+5]
            varhists = [hist[{"vars" : v}] for v in varnames]
            fig = plot_tools.makePlotWithRatioToRef(
                hists=[nominal] + varhists,
                labels=["Nominal"] + varnames,
                colors=["black"] + colors[0:len(varnames)],
                xlabel="p$_{T}^{\ell\ell}$ (GeV)", 
                ylabel="Events/bin",
                rlabel="x/nominal",
                rrange=[0.95, 1.05],
                nlegcols=1,
                xlim=None, binwnorm=1.0, baseline=True, 
                ratio_legend=False,
                yerr=False,
                yerr_ratio=False,
                linewidth=1,
            )
            fig.savefig(f"{outfolder}/summary_{int(idxplot/5):02d}.pdf")
            plt.close(fig)

    
    

if __name__ == "__main__":
    main()
