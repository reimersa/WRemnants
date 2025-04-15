import math
import os

import hist

import narf
from utilities import common, parsing
from wremnants import helicity_utils, unfolding_tools
from wremnants.datasets.datagroups import Datagroups
from wremnants.datasets.dataset_tools import getDatasets
from wremnants.histmaker_tools import write_analysis_output
from wums import logging

analysis_label = Datagroups.analysisLabel(os.path.basename(__file__))
parser, initargs = parsing.common_parser(analysis_label)
args = parser.parse_args()

logger = logging.setup_logger(__file__, args.verbose, args.noColorLogger)

datasets = getDatasets(
    maxFiles=args.maxFiles,
    filt=args.filterProcs,
    excl=args.excludeProcs,
    nanoVersion="v9",
    base_path=args.dataPath,
    extended="msht20an3lo" not in args.pdfs,
    era=args.era,
)

# ROOT.ROOT.DisableImplicitMT()


def build_graph(df, dataset):
    logger.info(f"build graph for dataset: {dataset.name}")
    results = []

    if dataset.is_data:
        df = df.DefinePerSample("weight", "1.0")
    else:
        df = df.Define("weight", "std::copysign(1.0, genWeight)")
    weightsum = df.SumAndCount("weight")

    df = unfolding_tools.define_gen_level(
        df, dataset.name, ["prefsr"], mode=analysis_label
    )

    df = df.Alias("nominal_weight", "weight")

    weightsByHelicity_helper = helicity_utils.makehelicityWeightHelper(
        is_z=dataset.name == "ZmumuPostVFP",
        filename=f"{common.data_dir}/angularCoefficients/w_z_helicity_xsecs_scetlib_dyturboCorr_maxFiles_m1_unfoldingBinning.hdf5",
    )

    results.append(
        df.HistoBoost(
            "massVgen",
            [
                hist.axis.Variable(
                    [
                        60,
                        70,
                        75,
                        78,
                        80,
                        82,
                        84,
                        85,
                        86,
                        87,
                        88,
                        89,
                        90,
                        91,
                        92,
                        93,
                        94,
                        95,
                        96,
                        97,
                        98,
                        100,
                        102,
                        105,
                        110,
                        120,
                    ],
                    name="massVgen",
                )
            ],
            ["massVgen"],
        )
    )
    results.append(
        df.HistoBoost(
            "absYVgen",
            [hist.axis.Variable([0, 0.35, 0.7, 1.1, 1.5, 2.5], name="absYVgen")],
            ["absYVgen"],
        )
    )
    results.append(
        df.HistoBoost(
            "ptVgen",
            [
                hist.axis.Variable(
                    [
                        0,
                        2,
                        3,
                        4,
                        5,
                        6,
                        7,
                        8,
                        9,
                        10,
                        11,
                        13,
                        15,
                        17,
                        20,
                        23,
                        27,
                        33,
                        100,
                    ],
                    name="ptVgen",
                )
            ],
            ["ptVgen"],
        )
    )
    results.append(
        df.HistoBoost(
            "chargeVgen",
            [
                hist.axis.Variable(
                    [-2.5, -1.5, -1.1, -0.35, 0, 0.35, 0.7, 1.1, 1.5, 2.5],
                    name="chargVgen",
                )
            ],
            ["chargeVgen"],
        )
    )

    df = df.Define("costheta", "csSineCosThetaPhigen.costheta")
    df = df.Define("phi", "csSineCosThetaPhigen.phi()")

    axis_cosThetaStarll = hist.axis.Variable(
        [-1, -0.56, -0.375, -0.19, 0.0, 0.19, 0.375, 0.56, 1.0],
        name="cosThetaStarll",
        underflow=False,
        overflow=False,
    )
    axis_phiStarll = hist.axis.Variable(
        [-math.pi, -2.27, -1.57, -0.87, 0, 0.87, 1.57, 2.27, math.pi],
        name="phiStarll",
        underflow=False,
        overflow=False,
    )
    results.append(
        df.HistoBoost(
            "csSineCosThetaPhigen",
            [axis_cosThetaStarll, axis_phiStarll],
            ["costheta", "phi"],
        )
    )

    df = df.Define(
        "helWeight_tensor",
        weightsByHelicity_helper,
        [
            "massVgen",
            "absYVgen",
            "ptVgen",
            "chargeVgen",
            "csSineCosThetaPhigen",
        ],
    )
    df = df.Define(
        "nominal_weight_helicity",
        "wrem::scalarmultiplyHelWeightTensor(nominal_weight, helWeight_tensor)",
    )

    from wremnants.helicity_utils import axis_helicity_multidim

    unfolding_axes = [
        hist.axis.Variable(
            [0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 15, 17, 20, 23, 27, 33, 100],
            name="ptVGen",
        ),
        hist.axis.Variable([0, 0.35, 0.7, 1.1, 1.5, 2.5], name="absYVGen"),
    ]
    unfolding_cols = ["prefsrV_pt", "prefsrV_absY"]

    results.append(
        df.HistoBoost(
            "prefsr",
            unfolding_axes,
            [*unfolding_cols, "nominal_weight_helicity"],
            tensor_axes=[axis_helicity_multidim],
        )
    )

    # look at polynomials and Ais
    df = df.Define(
        "angular_tensor",
        "wrem::csAngularFactors(csSineCosThetaPhigen)",
    )
    results.append(
        df.HistoBoost(
            "polynomials",
            [axis_cosThetaStarll, axis_phiStarll],
            ["costheta", "phi", "angular_tensor"],
            tensor_axes=[axis_helicity_multidim],
        )
    )

    return results, weightsum


logger.debug(f"Datasets are {[d.name for d in datasets]}")
resultdict = narf.build_and_run(datasets[::-1], build_graph)

write_analysis_output(
    resultdict, f"{os.path.basename(__file__).replace('py', 'hdf5')}", args
)
