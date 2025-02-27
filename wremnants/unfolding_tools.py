from copy import deepcopy

import hist
import numpy as np

from utilities import common
from wremnants import syst_tools, theory_corrections, theory_tools, theoryAgnostic_tools
from wums import logging

logger = logging.child_logger(__name__)


def add_out_of_acceptance(datasets, group, newGroupName=None):
    # Copy datasets from specified group to make out of acceptance contribution
    datasets_ooa = []
    for dataset in datasets:
        if dataset.group == group:
            ds = deepcopy(dataset)

            if newGroupName is None:
                ds.group = ds.group + "OOA"
            else:
                ds.group = newGroupName
            ds.out_of_acceptance = True

            datasets_ooa.append(ds)

    return datasets + datasets_ooa


def define_gen_level(df, dataset_name, gen_levels=["prefsr", "postfsr"], mode="w_mass"):
    # gen level definitions
    known_levels = ["prefsr", "postfsr"]
    if any(g not in known_levels for g in gen_levels):
        raise ValueError(
            f"Unknown gen level in '{gen_levels}'! Supported gen level definitions are '{known_levels}'."
        )

    singlelep = mode[0] == "w" or "wlike" in mode

    if "prefsr" in gen_levels:
        df = theory_tools.define_prefsr_vars(df)

        # # needed for fiducial phase space definition
        df = df.Alias("prefsrV_mass", "massVgen")
        df = df.Alias("prefsrV_pt", "ptVgen")
        df = df.Alias("prefsrV_absY", "absYVgen")
        df = df.Alias("prefsrV_charge", "chargeVgen")

        if singlelep:
            df = df.Alias("prefsr_mT", "mTVgen")

        if mode[0] == "w":
            df = df.Define("prefsrLep_pt", "chargeVgen < 0 ? genl.pt() : genlanti.pt()")
            df = df.Define(
                "prefsrLep_absEta",
                "chargeVgen < 0 ? std::fabs(genl.eta()) : std::fabs(genlanti.eta())",
            )
            df = df.Alias("prefsrLep_charge", "chargeVgen")
        else:
            df = df.Define("prefsrLep_pt", "event % 2 == 0 ? genl.pt() : genlanti.pt()")
            df = df.Define(
                "prefsrLep_absEta",
                "event % 2 == 0 ? std::fabs(genl.eta()) : std::fabs(genlanti.eta())",
            )
            df = df.Define(
                "prefsrOtherLep_pt", "event % 2 == 0 ? genlanti.pt() : genl.pt()"
            )
            df = df.Define(
                "prefsrOtherLep_absEta",
                "event % 2 == 0 ? std::fabs(genlanti.eta()) : std::fabs(genl.eta())",
            )
            if "wlike" in mode:
                df = df.Define("prefsrLep_charge", "event % 2 == 0 ? -1 : 1")

    if "postfsr" in gen_levels:
        df = theory_tools.define_postfsr_vars(df, mode=mode)

        if singlelep:
            df = df.Alias("postfsrV_mT", "postfsrMT")
        else:
            df = df.Alias("postfsrV_mass", "postfsrMV")
            df = df.Alias("postfsrV_absY", "postfsrabsYV")

        df = df.Alias("postfsrV_pt", "postfsrPTV")
        df = df.Alias("postfsrV_charge", "postfsrChargeV")

    return df


def select_fiducial_space(
    df, gen_level, select=True, accept=True, mode="w_mass", **kwargs
):
    # Define a fiducial phase space and if select=True, either select events inside/outside
    # accept = True: select events in fiducial phase space
    # accept = False: reject events in fiducial pahse space

    selmap = {
        x: None
        for x in [
            "pt_min",
            "pt_max",
            "abseta_max",
            "mass_min",
            "mass_max",
            "mtw_min",
        ]
    }

    selections = kwargs.get("selections", [])[:]
    fiducial = kwargs.get("fiducial")
    if fiducial:
        logger.info(
            f"Using default fiducial settings for selection {fiducial} for analysis {mode}"
        )
        if fiducial not in ["inclusive", "masswindow"]:
            # Use unfolding values in gen script
            selmap["pt_min"], selmap["pt_max"] = common.get_default_ptbins(
                mode, gen="vgen" in mode
            )[1:]
            selmap["abseta_max"] = common.get_default_etabins(mode)[-1]
            if mode[0] == "w" or "wlike" in mode:
                selmap["mtw_min"] = common.get_default_mtcut(mode)
        elif fiducial == "masswindow" and mode[0] == "z":
            selmap["mass_min"], selmap["mass_max"] = common.get_default_mz_window()
    else:
        for k in selmap.keys():
            selmap[k] = kwargs.get(k)

    if selmap["abseta_max"] is not None:
        selections.append(f"{gen_level}Lep_absEta < {selmap['abseta_max']}")
        if mode[0] == "z":
            selections.append(f"{gen_level}OtherLep_absEta < {selmap['abseta_max']}")

    if selmap["pt_min"] is not None:
        if "gen" in mode or "dilepton" in mode:
            selections.append(f"{gen_level}Lep_pt > {selmap['pt_min']}")
        if mode[0] == "z":
            selections.append(f"{gen_level}OtherLep_pt > {selmap['pt_min']}")

    if selmap["pt_max"] is not None:
        if "gen" in mode or "dilepton" in mode:
            # Don't place explicit cut on lepton pT for unfolding of W/W-like, but do for gen selection
            selections.append(f"{gen_level}Lep_pt < {selmap['pt_max']}")
        if mode[0] == "z":
            selections.append(f"{gen_level}OtherLep_pt < {selmap['pt_max']}")

    if selmap["mass_min"] is not None:
        selections.append(f"{gen_level}V_mass > {selmap['mass_min']}")

    if selmap["mass_max"] is not None:
        selections.append(f"{gen_level}V_mass < {selmap['mass_max']}")

    if selmap["mtw_min"] is not None:
        selections.append(f"{gen_level}V_mT > {selmap['mtw_min']}")

    selection = " && ".join(selections)

    if selection:
        df = df.Define(f"{gen_level}_acceptance", selection)
        logger.info(f"Applying fiducial selection '{selection}'")
    else:
        df = df.DefinePerSample(f"{gen_level}_acceptance", "true")

    if select and accept:
        logger.debug("Select events in fiducial phase space")
        df = df.Filter(f"{gen_level}_acceptance")
    elif select:
        logger.debug("Reject events in fiducial phase space")
        df = df.Filter(f"{gen_level}_acceptance == 0")

    return df


def add_xnorm_histograms(
    results,
    df,
    args,
    dataset_name,
    corr_helpers,
    qcdScaleByHelicity_helper,
    unfolding_axes,
    unfolding_cols,
    base_name="xnorm",
    add_helicity_axis=False,
):
    # add histograms before any selection
    df_xnorm = df
    df_xnorm = df_xnorm.DefinePerSample("exp_weight", "1.0")

    df_xnorm = theory_tools.define_theory_weights_and_corrs(
        df_xnorm, dataset_name, corr_helpers, args
    )

    df_xnorm = df_xnorm.Define("xnorm", "0.5")

    axis_xnorm = hist.axis.Regular(
        1, 0.0, 1.0, name="count", underflow=False, overflow=False
    )

    xnorm_axes = [axis_xnorm, *unfolding_axes]
    xnorm_cols = ["xnorm", *unfolding_cols]

    if add_helicity_axis:
        df_xnorm = theoryAgnostic_tools.define_helicity_weights(
            df_xnorm,
            filename=f"{common.data_dir}/angularCoefficients/w_z_moments_unfoldingBinning.hdf5",
        )

        from wremnants.helicity_utils import axis_helicity_multidim

        results.append(
            df_xnorm.HistoBoost(
                base_name,
                xnorm_axes,
                [*xnorm_cols, "nominal_weight_helicity"],
                tensor_axes=[axis_helicity_multidim],
            )
        )
    else:
        results.append(
            df_xnorm.HistoBoost(base_name, xnorm_axes, [*xnorm_cols, "nominal_weight"])
        )

    syst_tools.add_theory_hists(
        results,
        df_xnorm,
        args,
        dataset_name,
        corr_helpers,
        qcdScaleByHelicity_helper,
        xnorm_axes,
        xnorm_cols,
        base_name=base_name,
        addhelicity=add_helicity_axis,
        nhelicity=9,
    )


def reweight_to_fitresult(
    fitresult, axes, poi_type="nois", cme=13, process="Z", expected=False, flow=True
):
    # requires fitresult generated from 'fitresult_pois_to_hist.py'
    histname = "hist_" + "_".join([a.name for a in axes])
    if expected:
        histname += "_expected"

    import pickle

    with open(fitresult, "rb") as f:
        r = pickle.load(f)
        if process == "W":
            corrh_0 = r["results"][poi_type][f"chan_{str(cme).replace('.','p')}TeV"][
                "W_qGen0"
            ][histname]
            corrh_1 = r["results"][poi_type][f"chan_{str(cme).replace('.','p')}TeV"][
                "W_qGen1"
            ][histname]
        else:
            corrh = r["results"][poi_type][f"chan_{str(cme).replace('.','p')}TeV"][
                process
            ][histname]

    slices = [slice(None) for i in range(len(axes))]

    if "qGen" not in [a.name for a in axes]:
        # CorrectionsTensor needs charge axis
        if process == "Z":
            axes.append(hist.axis.Regular(1, -1, 1, name="chargeVGen", flow=False))
            slices.append(np.newaxis)
            values = corrh.values(flow=flow)
        elif process == "W":
            axes.append(hist.axis.Regular(2, -2, 2, name="chargeVGen", flow=False))
            slices.append(slice(None))
            values = np.stack(
                [corrh_0.values(flow=flow), corrh_1.values(flow=flow)], axis=-1
            )

    ch = hist.Hist(*axes, hist.axis.Regular(1, 0, 1, name="vars", flow=False))
    slices.append(np.newaxis)

    ch = theory_corrections.set_corr_ratio_flow(ch)
    ch.values(flow=flow)[...] = values[*slices]

    logger.debug(f"corrections from fitresult: {values}")

    from wremnants.correctionsTensor_helper import makeCorrectionsTensor

    return makeCorrectionsTensor(ch)
