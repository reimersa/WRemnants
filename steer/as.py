import os
from dataclasses import dataclass, field
from enum import Enum









def main():


    outfolder_base = "/scratch/submit/cms/areimers/wmass"
    plotfolder_base = "/work/submit/areimers/wmass/plots"
    wremnants_folder = "/work/submit/areimers/wmass/WRemnants"

    max_threads = 250

    corrs_and_postfixes_and_legendnames = [
        ("scetlib_dyturbo", "FixedAlphaS", "N$^{3{+}0}$LL+NNLO\n(nominal)"),

        # Theory corrections with new NP model. 1) that exactly reproduces the old central values and approximately the old variations. 2) With lattice-motivated central values and lattice-constrained uncertainties
        ("scetlib_dyturbo_NewNPModel_OldValsAndVars", "FixedAlphaS", "N$^{3{+}0}$LL+NNLO (new NP, old vals/vars)"),
        ("scetlib_dyturbo_NewNPModel_LatticeValsAndVars", "FixedAlphaS", "N$^{3{+}0}$LL+NNLO (new NP, lattice vals/vars)"),

    ]


    modes = [Mode.as_dilepton]
    
    

    for mode in modes:
        for (theory_corr, postfix, legname) in corrs_and_postfixes_and_legendnames:
            runner = AnalysisRunner(mode=mode, max_threads=max_threads, outfolder_base=outfolder_base, plotfolder_base=plotfolder_base, wremnants_folder=wremnants_folder, theory_corr=theory_corr, postfix=postfix)

            # runner.run_histmaker()

            # runner.run_setup_rabbit(add_theory_stat_unc=False)
            # runner.run_setup_rabbit(add_theory_stat_unc=True)

            # runner.run_fit_rabbit(add_theory_stat_unc=False)
            # runner.run_fit_rabbit(add_theory_stat_unc=True)

            runner.plot_fit_rabbit(prefit=True, add_theory_stat_unc=False)
            runner.plot_fit_rabbit(prefit=False, add_theory_stat_unc=False) # postfit
            # runner.plot_fit_rabbit(prefit=True, add_theory_stat_unc=True)
            # runner.plot_fit_rabbit(prefit=False, add_theory_stat_unc=True)  # postfit

            # runner.plot_impacts(add_theory_stat_unc=False)
            # runner.plot_impacts(add_theory_stat_unc=True)













class Mode(Enum):
    as_dilepton = "as_dilepton"

mode_configs = {
    Mode.as_dilepton: {
        "exec_histmaker": "mz_dilepton.py",
        "args_histmaker": " --axes yll ptll",
        "outputfile_histmaker": "mz_dilepton",
        "subfolder_histmaker": "ForAlphaS",
        "args_setup_rabbit": " --fitvar ptll-yll --fitAlphaS",
        "fitinput_folder": "ZMassDilepton_ptll_yll",
        "fitinput_filename": "ZMassDilepton.hdf5",
        "args_unblind": "",
    },
}

def make_postfix_plot(nominal_corr, postfix, add_theory_stat_unc, mode):
    result = f"{nominal_corr}Corr"
    if postfix:
        result += f"_{postfix}"
    if add_theory_stat_unc:
        result += "_PerBinStatCorrUnc"
    return result


@dataclass
class AnalysisRunner:
    mode: Mode
    outfolder_base: str
    plotfolder_base: str
    wremnants_folder: str
    theory_corr: str
    postfix: str = ""
    theory_corrs: str = ""
    max_threads: int = -1
    np_model: str = "Delta_Lambda"
    
    arg_pseudodata: str = field(init=False)
    nominal_corr: str = field(init=False)
    postfix_histmaker: str = field(init=False)
    postfix_fit_plot: str = field(init=False)
    exec_histmaker: str = field(init=False)
    args_histmaker: str = field(init=False)
    outputfile_histmaker: str = field(init=False)
    args_setup_rabbit: str = field(init=False)
    fitinput_folder: str = field(init=False)
    fitinput_filename: str = field(init=False)
    args_unblind: str = field(init=False)

    def __post_init__(self):
        self.theory_corrs = f"'{self.theory_corr}' '{self.theory_corr}_pdfas'" if self.theory_corr else "'scetlib_dyturbo' 'scetlib_dyturboCT18Z_pdfas'"
        if self.theory_corr == "scetlib_dyturbo":
            self.theory_corrs = self.theory_corrs.replace("scetlib_dyturbo_pdfas", "scetlib_dyturboCT18Z_pdfas")
        if "scetlib_dyturbo" not in self.theory_corrs:
            self.theory_corrs += " 'scetlib_dyturbo' 'scetlib_dyturboCT18Z_pdfas'"
        self.nominal_corr = self.theory_corrs.split(" ")[0].strip("'")
        
        self.postfix_histmaker = f"{self.postfix}"
        self.postfix_fit_plot = f"{self.nominal_corr}Corr"
        if self.postfix:
            self.postfix_fit_plot += f"_{self.postfix}"
        self.arg_pseudodata = f" --pseudoData '{self.nominal_corr}Corr'" if "scetlib_dyturbo" not in self.nominal_corr else " --pseudoData 'uncorr'"

        if "NewNPModel" in self.theory_corr:
            self.np_model = "tanh2"

            if "LatticeValsAndVars" in self.theory_corr:
                self.np_model = "lattice"
        


        self.exec_histmaker = mode_configs[self.mode]["exec_histmaker"]
        self.args_histmaker = mode_configs[self.mode]["args_histmaker"]
        self.outputfile_histmaker = mode_configs[self.mode]["outputfile_histmaker"]
        self.subfolder_histmaker = mode_configs[self.mode]["subfolder_histmaker"]
        self.args_setup_rabbit = mode_configs[self.mode]["args_setup_rabbit"]
        self.fitinput_folder = mode_configs[self.mode]["fitinput_folder"]
        self.fitinput_filename = mode_configs[self.mode]["fitinput_filename"]
        self.args_unblind = mode_configs[self.mode]["args_unblind"]
        







        


    def run_histmaker(self):

        arg_maxthreads = f" -j {self.max_threads}" if self.max_threads > 0 else ""

        # produces {outfolder_base}/histmaker/mz_dilepton_{theory_corr}Corr_maxFiles_m1_{postfix}.hdf5
        if self.subfolder_histmaker:
            outfolder = os.path.join(self.outfolder_base, 'histmaker', self.subfolder_histmaker)
        else:
            outfolder = os.path.join(self.outfolder_base, 'histmaker')
        command = f"python scripts/histmakers/{self.exec_histmaker} -o '{outfolder}' --theoryCorr {self.theory_corrs} --theoryCorrStatUnc '{self.theory_corr}'{self.args_histmaker} --maxFiles '-1' -p {self.postfix_histmaker}{arg_maxthreads}"

        print(command)
        os.system(command)



    def run_setup_rabbit(self, add_theory_stat_unc=False):
        # produces {outfolder_base}/fitinputs/ZMassDilepton_ptll_{theoryCorr}Corr_{arg_postfix}.hdf5

        arg_postfix = self.postfix_fit_plot + "_AlphaS"
        arg_theorystatunc = ""
        if add_theory_stat_unc:
            arg_postfix += "_PerBinStatCorrUnc"
            arg_theorystatunc = "--addTheoryCorrStatUnc"
        
        if self.subfolder_histmaker:
            infolder = os.path.join(self.outfolder_base, 'histmaker', self.subfolder_histmaker)
            outfolder = os.path.join(self.outfolder_base, 'fitinputs', self.subfolder_histmaker)
        else:
            infolder = os.path.join(self.outfolder_base, 'histmaker')
            outfolder = os.path.join(self.outfolder_base, 'fitinputs')


        command = f"python scripts/rabbit/setupRabbit.py -i '{os.path.join(infolder, f'{self.outputfile_histmaker}_{self.theory_corr}Corr_maxFiles_m1_{self.postfix_histmaker}.hdf5')}' --npUnc '{self.np_model}' -o '{outfolder}' --realData{self.args_setup_rabbit} -p {arg_postfix}{self.arg_pseudodata} {arg_theorystatunc}"

        print(command)
        os.system(command)

    def run_fit_rabbit(self, add_theory_stat_unc=False):
        # produces {outfolder_base}/fitresults/ZMassDilepton_ptll_{arg_postfix}/fitresults_data.hdf5

        arg_postfix = self.postfix_fit_plot + "_AlphaS"
        if add_theory_stat_unc:
            arg_postfix += "_PerBinStatCorrUnc"
        
        if self.subfolder_histmaker:
            infolder = os.path.join(self.outfolder_base, 'fitinputs', self.subfolder_histmaker)
            outfolder = os.path.join(self.outfolder_base, 'fitresults', self.subfolder_histmaker)
        else:
            infolder = os.path.join(self.outfolder_base, 'fitinputs')
            outfolder = os.path.join(self.outfolder_base, 'fitresults')


        command = f"{self.wremnants_folder}/rabbit/bin/rabbit_fit.py '{os.path.join(infolder, f'{self.fitinput_folder}_{arg_postfix}', self.fitinput_filename)}' --globalImpacts --computeHistErrors --computeVariations --saveHists --saveHistsPerProcess -m Basemodel -m Project ch0 ptll --doImpacts -t -1 --postfix asimov -o '{os.path.join(outfolder, f'{self.fitinput_folder}_{arg_postfix}')}'{self.args_unblind}"

        print(command)
        os.system(command)

        command = f"{self.wremnants_folder}/rabbit/bin/rabbit_fit.py '{os.path.join(infolder, f'{self.fitinput_folder}_{arg_postfix}', self.fitinput_filename)}' --globalImpacts --computeHistErrors --computeVariations --saveHists --saveHistsPerProcess -m Basemodel -m Project ch0 ptll --doImpacts -t 0 --postfix data -o '{os.path.join(outfolder, f'{self.fitinput_folder}_{arg_postfix}')}'{self.args_unblind}"

        print(command)
        os.system(command)




    def plot_fit_rabbit(self, prefit=True, add_theory_stat_unc=False):
        # produces {outfolder_base}/fitresults/ZMassDilepton_ptll_{arg_postfix}/fitresults_data.hdf5

        arg_postfix = self.postfix_fit_plot + "_AlphaS"
        if add_theory_stat_unc:
            arg_postfix += "_PerBinStatCorrUnc"
        
        if self.subfolder_histmaker:
            infolder = os.path.join(self.outfolder_base, 'fitresults', self.subfolder_histmaker)
            outfolder = os.path.join(self.plotfolder_base, 'fitresults', self.subfolder_histmaker)
        else:
            infolder = os.path.join(self.outfolder_base, 'fitresults')
            outfolder = os.path.join(self.plotfolder_base, 'fitresults')

        arg_prefit = " --prefit" if prefit else ""

        if prefit:
            arg_rrange = " --rrange 0.85 1.15"
        else: 
            arg_rrange = " --rrange 0.97 1.03"


        if self.mode in [Mode.as_dilepton]:
            # no var
            arg_variations = ""
            
            # alpha_s var
            # arg_variations = " --varNames pdfAlphaS --varLabel '$\\alpha_\\mathrm{S}{\\pm}1\\sigma$'"

            # # Gamma NP vars
            # if self.theory_corr == "scetlib_dyturbo_NewNPModel_LatticeValsAndVars":
            #     arg_variations = " --varNames scetlibNPgammaEigvar1 scetlibNPgammaEigvar2 scetlibNPgammaEigvar3 --varLabel 'SCETLib eig. 1' 'SCETLib eig. 3' 'SCETLib eig. 3' --showVariations both --lowerLegCols 4 --lowerLegPos 'upper right'"
            # else:
            #     arg_variations = " --varNames scetlibNPgamma --varLabel 'SCETLib $\\gamma$' --showVariations both --lowerLegCols 4 --lowerLegPos 'upper right'"

            # Lambda NP vars
            # if "NewNPModel" in self.theory_corr:
            #     arg_variations = " --varNames chargeVgenNP0scetlibNPZlambda2 chargeVgenNP0scetlibNPZlambda4 chargeVgenNP0scetlibNPZdelta_lambda2 --varLabel 'SCETLib $\\Lambda^{2}$' 'SCETLib $\\Lambda^{4}$' 'SCETLib $\\Delta\\Lambda^{2}$' --showVariations both --lowerLegCols 4 --lowerLegPos 'upper right'"
            # else:
            #     arg_variations = " --varNames chargeVgenNP0scetlibNPZLambda2 chargeVgenNP0scetlibNPZLambda4 chargeVgenNP0scetlibNPZDelta_Lambda2 --varLabel 'SCETLib $\\Lambda^{2}$' 'SCETLib $\\Lambda^{4}$' 'SCETLib $\\Delta\\Lambda^{2}$' --showVariations both --lowerLegCols 4 --lowerLegPos 'upper right'"

            # some of the largest PDF vars
            # arg_variations = " --varNames pdf14CT18ZSymAvg pdf26CT18ZSymAvg pdf27CT18ZSymAvg --varLabel 'PDF (14) [avg.]' 'PDF (26) [avg.]' 'PDF (27) [avg.]' --showVariations both --lowerLegCols 4 --lowerLegPos 'upper right'"

        command = f"{self.wremnants_folder}/rabbit/bin/rabbit_plot_hists.py --config utilities/styles/styles.py '{os.path.join(infolder, f'{self.fitinput_folder}_{arg_postfix}', f'fitresults_asimov.hdf5')}'  --title CMS --titlePos 1 --subtitle WiP -o '{os.path.join(outfolder, f'{self.fitinput_folder}_{arg_postfix}_asimov')}' --legCols 1 --yscale 1.2{arg_prefit} --legCol 1 --legSize small --noData -m Basemodel -m Project ch0 ptll{arg_variations}{arg_rrange}"
        print(command)
        os.system(command)

        command = f"{self.wremnants_folder}/rabbit/bin/rabbit_plot_hists.py --config utilities/styles/styles.py '{os.path.join(infolder, f'{self.fitinput_folder}_{arg_postfix}', f'fitresults_data.hdf5')}'  --title CMS --titlePos 1 --subtitle WiP -o '{os.path.join(outfolder, f'{self.fitinput_folder}_{arg_postfix}_data')}' --legCols 1 --yscale 1.2{arg_prefit} --legCol 1 --legSize small -m Basemodel -m Project ch0 ptll{arg_variations}{arg_rrange}"
        print(command)
        os.system(command)

   
    def plot_impacts(self, add_theory_stat_unc=False):

        arg_postfix = self.postfix_fit_plot + "_AlphaS"
        if add_theory_stat_unc:
            arg_postfix += "_PerBinStatCorrUnc"
        
        if self.subfolder_histmaker:
            infolder = os.path.join(self.outfolder_base, 'fitresults', self.subfolder_histmaker)
            outfolder = os.path.join(self.plotfolder_base, 'fitresults', self.subfolder_histmaker)
        else:
            infolder = os.path.join(self.outfolder_base, 'fitresults')
            outfolder = os.path.join(self.plotfolder_base, 'fitresults')

        postfix_fitresults_file = "asimov" if self.mode in [Mode.as_dilepton] else "data"

        for postfix_fitresults_file in ["asimov", "data"]:

            # traditional impacts
            command = f"{self.wremnants_folder}/rabbit/bin/rabbit_plot_pulls_and_impacts.py '{os.path.join(infolder, f'{self.fitinput_folder}_{arg_postfix}', f'fitresults_{postfix_fitresults_file}.hdf5')}' --title CMS --subtitle WiP --postfix {arg_postfix} --showNumbers --diffPullAsym --pullrange '2.1' --config 'utilities/styles/styles.py' --oneSidedImpacts --grouping max --otherExtensions pdf png -n 50 --scaleImpacts 1.5 -o '{os.path.join(outfolder, f'{self.fitinput_folder}_{arg_postfix}_{postfix_fitresults_file}')}'"
            print(command)
            os.system(command)

            # global impacts
            command = f"{self.wremnants_folder}/rabbit/bin/rabbit_plot_pulls_and_impacts.py '{os.path.join(infolder, f'{self.fitinput_folder}_{arg_postfix}', f'fitresults_{postfix_fitresults_file}.hdf5')}' --globalImpacts --title CMS --subtitle WiP --postfix {arg_postfix} --showNumbers --diffPullAsym --pullrange '2.1' --config 'utilities/styles/styles.py' --oneSidedImpacts --grouping max --otherExtensions pdf png -n 50 --scaleImpacts 1.5 -o '{os.path.join(outfolder, f'{self.fitinput_folder}_{arg_postfix}_{postfix_fitresults_file}')}'"
            print(command)
            os.system(command)

            # pulls
            command = f"{self.wremnants_folder}/rabbit/bin/rabbit_plot_pulls_and_impacts.py '{os.path.join(infolder, f'{self.fitinput_folder}_{arg_postfix}', f'fitresults_{postfix_fitresults_file}.hdf5')}' --title CMS --subtitle WiP --postfix {arg_postfix} --showNumbers --diffPullAsym --pullrange '2.1' --config 'utilities/styles/styles.py' --oneSidedImpacts --grouping max --otherExtensions pdf png -n 50 --scaleImpacts 1.5 -o '{os.path.join(outfolder, f'{self.fitinput_folder}_{arg_postfix}_{postfix_fitresults_file}')}' --noImpacts"
            print(command)
            os.system(command)


def mass_summary_modeling(mode, corrs_and_postfixes_and_legendnames, wremnants_folder, outfolder_base, plotfolder_base, add_theory_stat_unc=False):
    
    arg_postfixes = "--postfixes"
    arg_namesLegend = "--namesLegend"
    for corr, postfix, legname in corrs_and_postfixes_and_legendnames:
        arg_postfix = make_postfix_plot(nominal_corr=corr, postfix=postfix, add_theory_stat_unc=add_theory_stat_unc, mode=mode)
        arg_postfixes += f" '_{arg_postfix}'"
        arg_namesLegend += f" '{legname}'"

    arg_postfix = " -p 'PerBinStatCorrUnc'" if add_theory_stat_unc else ""


    # To make plot locally
    command = f"python {wremnants_folder}/scripts/plotting/mass_summary_modeling.py -r '{outfolder_base}/fitresults/{mode_configs[mode]['fitinput_folder']}{'{postfix}'}/fitresults_data.hdf5' --print --diffToCentral {arg_postfixes} {arg_namesLegend} -o '{os.path.join(plotfolder_base, 'fitresults')}' -f ''{arg_postfix} --cmsDecor ' '"
    print(command)
    os.system(command)

    # To copy to paper folder on LXPLUS eos
    if mode == Mode.mz_wlike:
        eos_subfolder = "Z_Wlike" 
    elif mode == Mode.mw:
        eos_subfolder = "W" 
    elif mode == Mode.mwminus:
        eos_subfolder = "Wm" 
    command = f"python {wremnants_folder}/scripts/plotting/mass_summary_modeling.py -r '{outfolder_base}/fitresults/{mode_configs[mode]['fitinput_folder']}{'{postfix}'}/fitresults_data.hdf5' --print --diffToCentral {arg_postfixes} {arg_namesLegend} -o '/eos/user/c/cmsmwbot/www/WMassAnalysis/PlotsForPaper/NatureResubmission1/{eos_subfolder}/' -f ''{arg_postfix} --cmsDecor ' ' --eoscp"
    
    
    # print(command)
    # os.system(command)
        


if __name__ == "__main__":
    main()





