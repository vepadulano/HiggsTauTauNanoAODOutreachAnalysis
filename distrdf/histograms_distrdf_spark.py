# Implementation of the histogramming step of the analysis
#
# The histogramming step produces histograms for each variable in the dataset
# and for each physics process resulting into the final state with a muon and a
# tau. Then, the resulting histograms are passed to the plotting step, which
# combines the histograms so that we can study the physics of the decay.
import os

import pyspark
import ROOT

ROOT.gROOT.SetBatch(True)
RDataFrame = ROOT.RDF.Experimental.Distributed.Spark.RDataFrame

# Names of the datasets to be found in the base path and processed for the
# analysis
samplesandlabels = [
    ("GluGluToHToTauTau", "ggH"),
    ("VBF_HToTauTau", "qqH"),
    ("W1JetsToLNu", "W1J"),
    ("W2JetsToLNu", "W2J"),
    ("W3JetsToLNu", "W3J"),
    ("TTbar", "TT"),
    ("DYJetsToLL", "ZLL"),
    ("DYJetsToLL", "ZTT"),
    ("Run2012B_TauPlusX", "dataRunB"),
    ("Run2012C_TauPlusX", "dataRunC"),
]

# Declare the range of the histogram for each variable
#
# Each entry in the dictionary contains of the variable name as key and a tuple
# specifying the histogram layout as value. The tuple sets the number of bins,
# the lower edge and the upper edge of the histogram.
default_nbins = 30
histmodels = {
    "pt_1": (default_nbins, 17, 70),
    "pt_2": (default_nbins, 20, 70),
    "eta_1": (default_nbins, -2.1, 2.1),
    "eta_2": (default_nbins, -2.3, 2.3),
    "phi_1": (default_nbins, -3.14, 3.14),
    "phi_2": (default_nbins, -3.14, 3.14),
    "iso_1": (default_nbins, 0, 0.10),
    "iso_2": (default_nbins, 0, 0.10),
    "q_1": (2, -2, 2),
    "q_2": (2, -2, 2),
    "pt_met": (default_nbins, 0, 60),
    "phi_met": (default_nbins, -3.14, 3.14),
    "m_1": (default_nbins, 0, 0.2),
    "m_2": (default_nbins, 0, 2),
    "mt_1": (default_nbins, 0, 100),
    "mt_2": (default_nbins, 0, 100),
    "dm_2": (11, 0, 11),
    "m_vis": (default_nbins, 20, 140),
    "pt_vis": (default_nbins, 0, 60),
    "jpt_1": (default_nbins, 30, 70),
    "jpt_2": (default_nbins, 30, 70),
    "jeta_1": (default_nbins, -4.7, 4.7),
    "jeta_2": (default_nbins, -4.7, 4.7),
    "jphi_1": (default_nbins, -3.14, 3.14),
    "jphi_2": (default_nbins, -3.14, 3.14),
    "jm_1": (default_nbins, 0, 20),
    "jm_2": (default_nbins, 0, 20),
    "jbtag_1": (default_nbins, 0, 1.0),
    "jbtag_2": (default_nbins, 0, 1.0),
    "npv": (25, 5, 30),
    "njets": (5, 0, 5),
    "mjj": (default_nbins, 0, 400),
    "ptjj": (default_nbins, 0, 200),
    "jdeta": (default_nbins, -9.4, 9.4),
}


def filterGenMatch(df, label):
    """
    Apply a selection based on generator information about the tau. See the
    skimming step for further details about this variable.
    """
    if label == "ZTT":
        return df.Filter("gen_match == true")
    elif label == "ZLL":
        return df.Filter("gen_match == false")
    else:
        return df


def histos(df, label):
    """
    Main function of the histogramming step
    The function loops over the outputs from the skimming step and produces the
    required histograms for the final plotting.
    Note that we perform a set of secondary selections on the skimmed dataset.
    First, we perform a second reduction with the baseline selection to a
    signal-enriched part of the dataset. Second, we select besides the signal
    region a control region which is used to estimate the contribution of QCD
    events producing the muon-tau pair in the final state.
    """

    # apply baseline selection
    baseline_selection = df.Filter(
        "mt_1<30",
        "Muon transverse mass cut for W+jets suppression")\
        .Filter("iso_1<0.1", "Require isolated muon for signal region")

    # Book histograms for the signal region
    signal_df = baseline_selection.Filter(
        "q_1*q_2<0", "Require opposited charge for signal region")
    signal_df = filterGenMatch(signal_df, label)
    signal_hists = {
        variable: signal_df.Histo1D(
            ROOT.RDF.TH1DModel(variable, variable, hmodel[0],
                               hmodel[1], hmodel[2]),
            variable, "weight")
        for variable, hmodel in histmodels.items()
    }

    # Book histograms for the control region used to estimate the QCD
    # contribution
    control_df = baseline_selection.Filter(
        "q_1*q_2>0", "Control region for QCD estimation")
    control_df = filterGenMatch(control_df, label)
    control_hists = {
        variable: control_df.Histo1D(
            ROOT.RDF.TH1DModel(variable, variable, hmodel[0],
                               hmodel[1], hmodel[2]),
            variable, "weight")
        for variable, hmodel in histmodels.items()
    }

    # Write histograms to output file
    for variable, hist in signal_hists.items():
        hist.SetName("{}_{}".format(label, variable))
        hist.Write()

    for variable, hist in control_hists.items():
        hist.SetName("{}_{}_cr".format(label, variable))
        hist.Write()


def main():
    """
    Main function of the histogramming step

    The function loops over the outputs from the skimming step and produces the
    required histograms for the final plotting.
    Note that we perform a set of secondary selections on the skimmed dataset.
    First, we perform a second reduction with the baseline selection to a
    signal-enriched part of the dataset. Second, we select besides the signal
    region a control region which is used to estimate the contribution of QCD
    events producing the muon-tau pair in the final state.
    """
    print("Starting HTT histogramming")
    histowatch = ROOT.TStopwatch()

    # Create output file
    tfile = ROOT.TFile("histograms.root", "RECREATE")

    # Example configuration with a YARN cluster, change according to needs
    conf = {
        "spark.master": "yarn",
        "spark.executorEnv.LD_LIBRARY_PATH": os.environ["LD_LIBRARY_PATH"],
        "spark.yarn.queue": "htt",
    }
    sconf = pyspark.SparkConf().setAll(conf.items())
    sc = pyspark.SparkContext(conf=sconf)

    # Folder where the snapshotted data is stored
    outbasepath = "skimdata/"

    for sample, label in samplesandlabels:
        print("\tStarting histo sample {} with label {}".format(sample, label))
        labelwatch = ROOT.TStopwatch()

        filenames = [outbasepath + filename for filename in os.listdir(outbasepath) if sample in filename]
        # Create RDataFrame
        df = RDataFrame("Events", filenames, npartitions=1000)

        # Create histograms
        histos(df, label)
        # Snapshot needed columns
        labelwatch.Stop()
        labeltime = round(labelwatch.RealTime(), 2)
        print("\tSample skimming ended in {} minutes".format(str(labeltime/60)))

    histowatch.Stop()
    histotime = round(histowatch.RealTime(), 2)
    print("HTT histogramming ended in {} minutes".format(str(histotime/60)))
    tfile.Close()


if __name__ == "__main__":
    main()
