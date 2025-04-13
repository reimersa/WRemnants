import hist
import numpy as np
import ROOT

from wremnants.correctionsTensor_helper import makeCorrectionsTensor
from wremnants.helicity_utils import axis_helicity, axis_helicity_multidim
from wremnants.theory_tools import helicity_xsec_to_angular_coeffs, scale_tensor_axes

###
# unit test for
# - polynomials
# - angular coefficients
# - helicity cross sections after reweighting
# - closure test by comparing input angular coefficients with angular coefficients from helicity cross sections after reweighting
###

# make some toy data
# Boson (p, y, m, q) with one entry per bin of correction
# original binning was: [0, 0.35, 0.7, 1.1, 1.5, 2.5]
y_centers = np.array([0.2, 0.5, 0.9, 1.3, 2.0], dtype=np.float64)
y_edges = np.append(y_centers, 3)

# original binning was: [0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 15, 17, 20, 23, 27, 33, 100]
p_edges = np.array(
    [0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 15, 17, 20, 23, 27, 33, 40],
    dtype=np.float64,
)
p_centers = p_edges[:-1] + np.diff(p_edges) / 2.0

y = np.tile(y_centers, (len(p_centers), 1)).T
p = np.tile(p_centers, (len(y_centers), 1))

m = 90 * np.ones(len(p_edges) - 1, dtype=np.float64)
q = np.zeros(len(p_edges) - 1, dtype=np.int32)

m = np.tile(m, (len(y_edges) - 1, 1))
q = np.tile(q, (len(y_edges) - 1, 1))

m = m.flatten()
q = q.flatten()
y = y.flatten()
p = p.flatten()

# CS variables flat in cosTheta* and phi*
# cosTheta*
cost_edges = np.linspace(-1, 1, 51, dtype=np.float64)
cost_centers = cost_edges[:-1] + np.diff(cost_edges) / 2.0

# phi*
phi_edges = np.linspace(-3.14, 3.14, 51, dtype=np.float64)
phi_centers = phi_edges[:-1] + np.diff(phi_edges) / 2.0

cost = cost_centers
phi = phi_centers

# weight vector to introduce polarization
theta = np.arccos(cost)

w = [
    np.outer(1 + cost * cost, np.ones_like(phi)),  # unpolarized
    np.outer(0.5 * (1 - 3 * cost * cost), np.ones_like(phi)),
    np.outer(np.sin(2 * theta), np.cos(phi)),
    np.outer(0.5 * np.sin(theta) ** 2, np.cos(2 * phi)),
    np.outer(np.sin(theta), np.cos(phi)),
    np.outer(cost, np.ones_like(phi)),
    np.outer(np.sin(theta) ** 2, np.sin(2 * phi)),
    np.outer(np.sin(2 * theta), np.sin(phi)),
    np.outer(np.sin(theta), np.sin(phi)),
]

# assume AIs with some pT dependence
a = [
    0 * p + 1.0,  # unpolarized always 1
    (p / 100) ** 2.0,  # assume a0 quadratic in pT
    p / 1000,  # assume a1 linear in pT
    (p / 100.0) ** 2,  # assume a2 quadratic in pT
    (p / 1000.0) ** 2,  # assume a3 quadratic in pT
    0 * p + 0.01,  # assume a4 independent of pT
    0 * p,  # assume a5 independent of pT
    0 * p,  # assume a6 independent of pT
    0 * p,  # assume a7 independent of pT
]

w = [np.outer(wi, ai) for wi, ai in zip(w, a)]
w = np.sum(w, axis=0)

# broadcase duplicate (p, y, m, q) to (phi,cost) space and vice versa
cost = np.outer(cost, np.ones_like(phi_centers))
phi = np.outer(np.ones_like(cost_centers), phi)

phi = phi.flatten()
cost = cost.flatten()

m = np.tile(m, (len(phi), 1))
q = np.tile(q, (len(phi), 1))
y = np.tile(y, (len(phi), 1))
p = np.tile(p, (len(phi), 1))

phi = np.tile(phi, (m.shape[-1], 1)).T
cost = np.tile(cost, (m.shape[-1], 1)).T

data = {
    k: v.flatten()
    for k, v in {
        "m": m,
        "y": y,
        "p": p,
        "q": q,
        "cost": cost,
        "phi": phi,
        "w": w,
    }.items()
}

scaleweight_tensor = np.ones((len(cost.flatten()), 8))
for i in range(8):
    data[f"s{i}"] = np.ones_like(data["m"], dtype=np.float32)

print("generate RDataFrame from toy data to make correction")
df = ROOT.RDF.FromNumpy(data)

df = df.Define("sint", "std::sqrt(1 - cost * cost)")
df = df.Define("sinp", "std::sin(phi)")
df = df.Define("cosp", "std::cos(phi)")

df = df.Define(
    "angles", "wrem::CSVars angles = {sint, cost, sinp, cosp}; return angles;"
)

df = df.Define("theory_weight_truncate", "10.")

df = df.Define("nominal_weight", "w")

# dummy scale weight vector
df = df.Define(
    "scaleweight_tensor",
    "ROOT::VecOps::RVec<float> res = {s0, s1, s2, s3, s4, s5, s6, s7}; return res;",
)
df = df.Define(
    "scaleWeights_tensor",
    f"wrem::makeScaleTensor(scaleweight_tensor, theory_weight_truncate);",
)

df = df.Define(
    "helicity_xsecs_scale_tensor",
    "wrem::makeHelicityMomentScaleTensor(angles, scaleWeights_tensor, nominal_weight)",
)

helicity_xsecs_scale = df.HistoBoost(
    f"nominal_gen_helicity_xsecs_scale",
    [
        hist.axis.Regular(1, 60.0, 120.0, name="massVgen"),
        hist.axis.Variable(y_edges, name="absYVgen", underflow=False),
        hist.axis.Variable(p_edges, name="ptVgen", underflow=False),
        hist.axis.Integer(0, 1, name="chargeVgen", underflow=False, overflow=False),
    ],
    ["m", "y", "p", "q", "helicity_xsecs_scale_tensor"],
    tensor_axes=[axis_helicity, *scale_tensor_axes],
    storage=hist.storage.Double(),
)

print("trigger processing")
h_helicity_xsecs_scales = helicity_xsecs_scale.GetValue()

# filename=f"{common.data_dir}/angularCoefficients/w_z_helicity_xsecs_scetlib_dyturboCorr_maxFiles_m1_unfoldingBinning.hdf5"

# weightsByHelicity_helper = helicity_utils.makehelicityWeightHelper(
#     is_z=True,
#     filename=filename
#     )


# generate helper as done in 'helicity_utils.makehelicityWeightHelper'
def make_angular_coefficient_helper(helper):
    # with h5py.File(filename, "r") as ff:
    #     out = input_tools.load_results_h5py(ff)

    # hist_helicity_xsec_scales = h_helicity_xsecs_scale["Z"]

    corrh = helicity_xsec_to_angular_coeffs(h_helicity_xsecs_scales)

    if "muRfact" in corrh.axes.name:
        corrh = corrh[
            {
                "muRfact": 1.0j,
            }
        ]
    if "muFfact" in corrh.axes.name:
        corrh = corrh[
            {
                "muFfact": 1.0j,
            }
        ]

    axes_names = ["massVgen", "absYVgen", "ptVgen", "chargeVgen", "helicity"]
    if not list(corrh.axes.name) == axes_names:
        raise ValueError(
            f"Axes [{corrh.axes.name}] are not the ones this functions expects ({axes_names})"
        )

    if np.count_nonzero(corrh[{"helicity": -1.0j}] == 0):
        raise ValueError(
            "Zeros in sigma UL for the angular coefficients will give undefined behaviour!"
        )
    # histogram has to be without errors to load the tensor directly
    corrh_noerrs = hist.Hist(*corrh.axes, storage=hist.storage.Double())
    corrh_noerrs.values(flow=True)[...] = corrh.values(flow=True)

    return makeCorrectionsTensor(corrh_noerrs, helper, tensor_rank=1)


print("Make helpers for reweighting to helicity cross sections")
angular_coefficients_helper = make_angular_coefficient_helper(
    ROOT.wrem.WeightByAngularCoefficientHelper
)
weightsByHelicity_helper = make_angular_coefficient_helper(
    ROOT.wrem.WeightByHelicityHelper
)

print("generate RDataFrame from toy data to evaluate correction")
data = {
    k: v.flatten()
    for k, v in {
        "m": m,
        "y": y,
        "p": p,
        "q": q,
        "cost": cost,
        "phi": phi,
        "w": w,
    }.items()
}

df = ROOT.RDF.FromNumpy(data)

df = df.Define("sint", "std::sqrt(1 - cost * cost)")
df = df.Define("sinp", "std::sin(phi)")
df = df.Define("cosp", "std::cos(phi)")

df = df.Define(
    "angles", "wrem::CSVars angles = {sint, cost, sinp, cosp}; return angles;"
)

df = df.Define("polynomialWeight_tensor", "wrem::csAngularFactors(angles)")

df = df.Define(
    "angularCoefficientsWeight_tensor",
    angular_coefficients_helper,
    [
        "m",
        "y",
        "p",
        "q",
    ],
)

df = df.Define(
    "helWeight_tensor",
    weightsByHelicity_helper,
    [
        "m",
        "y",
        "p",
        "q",
        "angles",
    ],
)

df = df.Define("weight_polynomial", f"w/{(len(p_centers) * len(y_centers))}")

df = df.Define(
    "nominal_weight_polynomial_tensor",
    "wrem::scalarmultiplyHelWeightTensor(weight_polynomial, polynomialWeight_tensor)",
)

df = df.Define("weight_ai", f"w/{(len(cost_centers) * len(phi_centers))}")

df = df.Define(
    "nominal_weight_ai_tensor",
    "wrem::scalarmultiplyHelWeightTensor(weight_ai, angularCoefficientsWeight_tensor)",
)

df = df.Define(
    "nominal_hel_weight_tensor",
    "wrem::scalarmultiplyHelWeightTensor(weight_ai, helWeight_tensor)",
)

h_pol = df.HistoBoost(
    "polynomials",
    [
        hist.axis.Variable(cost_edges, name="cost"),
        hist.axis.Variable(phi_edges, name="phi"),
    ],
    ["cost", "phi", "nominal_weight_polynomial_tensor"],
    tensor_axes=[axis_helicity_multidim],
)
h_ang = df.HistoBoost(
    "angularCoefficients",
    [hist.axis.Variable(p_edges, name="p"), hist.axis.Variable(y_edges, name="y")],
    ["p", "y", "nominal_weight_ai_tensor"],
    tensor_axes=[axis_helicity_multidim],
)
h_hel = df.HistoBoost(
    "helicityXsec",
    [hist.axis.Variable(p_edges, name="p"), hist.axis.Variable(y_edges, name="y")],
    ["p", "y", "nominal_hel_weight_tensor"],
    tensor_axes=[axis_helicity_multidim],
)

# trigger processing
h_pol = h_pol.GetValue()
h_ang = h_ang.GetValue()
h_hel = h_hel.GetValue()

# make some plots
outdir = (
    "/home/submit/david_w/public_html/AlphaS/250410_angularCoefficients/module_test/"
)
import matplotlib.pyplot as plt

### make plot for plynomials
# Create the plot
plt.figure()

# index shited by 1
for i in range(9):
    val = h_pol[{"helicitySig": i, "phi": hist.sum}].values() / len(phi_centers)
    plt.stairs(val, cost_edges, label=f"P_{i-1}" if i > 0 else f"P_UL")

plt.xlabel("cos(Theta)")
plt.ylabel(f"Polynomial")
plt.legend()
plt.grid(True)

# Save the figure
for ext in ["png", "pdf"]:
    plt.savefig(f"{outdir}/polynomials_cost.{ext}")

plt.close()

plt.figure()

# index shited by 1
for i in range(9):
    val = h_pol[{"helicitySig": i, "cost": hist.sum}].values() / len(cost_centers)
    plt.stairs(val, phi_edges, label=f"P_{i-1}" if i > 0 else f"P_UL")

plt.xlabel("phi")
plt.ylabel(f"Polynomial")
plt.legend()
plt.grid(True)

# Save the figure
for ext in ["png", "pdf"]:
    plt.savefig(f"{outdir}/polynomials_phi.{ext}")

plt.close()

### make plots for angular coefficients
colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
for i in range(1, 9):
    # Create the plot
    plt.figure()

    for j in range(0, 5):
        val = h_ang[{"helicitySig": i, "y": j}].values()
        plt.stairs(val, p_edges, label=f"|Y|={y_centers[j]}", color=colors[j])

        # angular coefficients from helicity cross sections via A_i = xsec_i / xsec_UL
        #   they should be the same -> closure test
        val_hel_UL = h_hel[{"helicitySig": 0, "y": j}].values()
        val_from_hel = h_hel[{"helicitySig": i, "y": j}].values() / val_hel_UL
        plt.plot(p_centers, val_from_hel, linestyle="", marker=".", color=colors[j])

    plt.xlabel("pT")
    plt.ylabel(f"Angular coefficient {i-1}")
    plt.legend()
    plt.grid(True)

    plt.xlim(0, 40)

    # Save the figure
    for ext in ["png", "pdf"]:
        plt.savefig(f"{outdir}/angular_coefficients_{i-1}.{ext}")

    plt.close()

### make plot for helicity cross sections
for i in range(0, 9):
    index = i - 1 if i > 0 else "UL"
    # Create the plot
    plt.figure()

    for j in range(0, 5):
        val = h_hel[{"helicitySig": i, "y": j}].values()
        plt.stairs(val, p_edges, label=f"|Y|={y_centers[j]}")

    plt.xlabel("pT")
    plt.ylabel(rf"Helcicity cross section $\sigma_{index}$")
    plt.legend()
    plt.grid(True)

    plt.xlim(0, 40)

    # Save the figure
    for ext in ["png", "pdf"]:
        plt.savefig(f"{outdir}/helicity_cross_section_{index}.{ext}")

    plt.close()
