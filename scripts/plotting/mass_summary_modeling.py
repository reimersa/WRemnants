import matplotlib.pyplot as plt # type: ignore
import pandas as pd # type: ignore
from matplotlib import ticker # type: ignore

from utilities import parsing
from utilities.io_tools import hepdata_tools, rabbit_input
from wums import output_tools, plot_tools

parser = parsing.plot_parser()
parser.add_argument(
    "-r",
    "--reffile",
    required=True,
    type=str,
    help="Combine fitresult file for nominal result",
)
parser.add_argument("--print", action="store_true", help="Print results")
parser.add_argument(
    "--diffToCentral", action="store_true", help="Show difference to central result"
)
parser.add_argument(
    "--saveForHepdata",
    action="store_true",
    help="Save output as ROOT to prepare HEPData",
)
parser.add_argument(
    "--postfixes",
    type=str,
    nargs="*",
    help="Postfixes of files to insert into the {} given in the args.reffile",
)
parser.add_argument(
    "--namesLegend",
    type=str,
    nargs="*",
    help="Strings to use in the legend for each p in args.postfixes (same length and order).",
)
args = parser.parse_args()

basename = args.reffile
isW = "WMass" in args.reffile
isWminus = False
if args.postfixes and isW:
    for p in args.postfixes:
        if not "_Wm" in p:
            isWminus = False
            break
        isWminus = True

if isW:
    if isWminus:
        additional_fits = ["/scratch/submit/cms/kdlong/CombineStudies/Unblinded/WMass_charge_eta_pt_dataPtllRwgt_Wm/fitresults.hdf5", "/scratch/submit/cms/kdlong/CombineStudies/Unblinded/Combination_WMassZMassDilepton_Wm/fitresults.hdf5", "/scratch/submit/cms/areimers/wmass/fitresults/WMass_charge_pt_eta_scetlib_dyturboCorr_FixedAlphaS_BinnedScale_Wm/fitresults_data.hdf5"]
        additional_legends = [r"$\mathit{p}_{T}^{\ell\ell}$ rwgt.," + " N$^{3{+}0}$LL unc.", "Combined " + r"$\mathit{p}_{T}^{\ell\ell}$" + " fit,\nN$^{3{+}0}$LL unc.", "N$^{3}$LL+NNLO,\n" + r"$\mathit{p}_{T}^{W}$"+"-binned scale unc."]
    else:
        additional_fits = ["/scratch/submit/cms/kdlong/CombineStudies/Unblinded/AltTheory/WMass_eta_pt_charge_dataPtllRwgt/fitresults.hdf5", "/scratch/submit/cms/areimers/wmass/fitresults/WMass_eta_pt_charge_CombinedPtll/fitresults.hdf5", "/scratch/submit/cms/kdlong/CombineStudies/Unblinded/AltTheory/WMass_eta_pt_charge_binnedScale/fitresults.hdf5"]
        additional_legends = [r"$\mathit{p}_{T}^{\ell\ell}$ rwgt.," + " N$^{3{+}0}$LL unc.", "Combined " + r"$\mathit{p}_{T}^{\ell\ell}$" + " fit,\nN$^{3{+}0}$LL unc.", "N$^{3}$LL+NNLO,\n" + r"$\mathit{p}_{T}^{W}$"+"-binned scale unc."] # 
else: # W-like
    additional_fits = ["/scratch/submit/cms/kdlong/CombineStudies/Unblinded/ZMassWLike_eta_pt_charge_dataPtllRwgt/fitresults.hdf5", "/scratch/submit/cms/areimers/wmass/fitresults/ZMassWLike_eta_pt_charge_scetlib_dyturboCorr_FixedAlphaS_BinnedScale/fitresults_data.hdf5"]
    additional_legends = [r"$\mathit{p}_{T}^{\ell\ell}$ rwgt.," + " N$^{3{+}0}$LL unc.", "N$^{3}$LL+NNLO,\n" + r"$\mathit{p}_{T}^{Z}$"+"-binned scale unc."]

dfs = rabbit_input.read_all_groupunc_df(
    [args.reffile.format(postfix=p) for p in args.postfixes] + additional_fits,
    names=args.namesLegend+additional_legends,
    uncs=["pTModeling"],
)

if isWminus:
    xlim = [80271, 80362]
elif isW:
    xlim = [80319, 80374]
else:
    xlim = [91120, 91205] if "flipEvenOdd" not in basename else [91170, 91290]

if args.print:
    for k, v in dfs.iterrows():
        print(v.iloc[0], round(v.iloc[1], 3), round(v.iloc[3], 3), round(v.iloc[2], 5))

central = dfs.iloc[0, :]

xlabel = r"$\mathit{m}_{" + ("W^{-}" if isWminus else "W" if isW else "Z") + "}$ (MeV)"

central_val = central["value"]
if args.diffToCentral:
    if args.saveForHepdata:
        # save also the original absolute value
        dfs["absolute_value"] = dfs["value"].values
    dfs["value"] -= central_val
    xlim = [xlim[0] - central_val, xlim[1] - central_val]
    central_val = 0
    xlabel = r"$\Delta$" + xlabel

legsize = 19
pt_size = 0.45
if isWminus:
    legsize = 19
    pt_size = 0.47

fig = plot_tools.make_summary_plot(
    central_val,
    central["err_total"],
    central["err_pTModeling"],
    args.namesLegend[0],
    dfs.iloc[1:, :],
    colors="auto",
    xlim=xlim,
    xlabel=xlabel,
    legend_loc="upper left",
    legtext_size=legsize,
    logoPos=0,
    cms_label=args.cmsDecor,
    lumi=16.8,
    padding=5,
    point_size=pt_size,
    width_scale=0.85,
)
ax = plt.gca()
ax.yaxis.set_major_locator(ticker.MultipleLocator(20))
if isW and not isWminus:
    ax.xaxis.set_major_locator(ticker.MultipleLocator(10))
else:
    ax.xaxis.set_major_locator(ticker.MultipleLocator(20))
ax.xaxis.set_minor_locator(ticker.MultipleLocator(5))
ax.xaxis.grid(False, which="both")
ax.yaxis.grid(False, which="both")

eoscp = output_tools.is_eosuser_path(args.outpath)
outdir = output_tools.make_plot_dir(args.outpath, args.outfolder, eoscp=eoscp)

outname = f"{'Wminusmass' if isWminus else "Wmass" if isW else 'Wlike'}_modeling_summary"
if args.postfix:
    outname += f"_{args.postfix}"

plot_tools.save_pdf_and_png(outdir, outname, fig)
output_tools.write_index_and_log(outdir, outname)

if args.saveForHepdata:
    column_labels = [xlabel, "Total uncertainty", "Model uncertainty"]
    if args.diffToCentral:
        column_labels.append(xlabel.replace(r"$\Delta$", ""))

    hepdata_tools.make_mass_summary_histogram(
        dfs, f"{outdir}/{outname}.root", column_labels
    )

if eoscp:
    output_tools.copy_to_eos(outdir, args.outpath, args.outfolder)
