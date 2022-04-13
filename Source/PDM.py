import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

from brian2 import *

epsilon = 1.0e-6

showGaussian = False
showPoisson = False
showNoise = False

# ==================================================================

def LinePlot(start, stop, step, f, xUnits, yUnits, *args, **kwargs):
    x = []
    y = []
    xEpsilon = epsilon * xUnits
    yEpsilon = epsilon * yUnits
    xRange = np.arange(start, stop, step)
    for xp in xRange:
        yp = f(xp)
        if len(x) == 0 or abs(xp - xRange[-1]) < xEpsilon:
            x.append(xp / xUnits)
            y.append(yp / yUnits)
        elif abs(yp - y[-1] * yUnits) > yEpsilon:
            if abs(xpp - x[-1] * xUnits) > xEpsilon:
                x.append(xpp / xUnits)
                y.append(ypp / yUnits)
            x.append(xp / xUnits)
            y.append(yp / yUnits)
        xpp = xp
        ypp = yp
    plt.plot(x, y, *args, **kwargs)

# ==================================================================

if __name__ == "__main__":

    print("=== Initialization ====================================")
    start_scope()

    # Simulation parameters
    simTime = 4*second
    activeTime = 2*second
    defaultclock.dt = 0.05*ms
    ratePeriod = 50*ms
    NA = 800
    NB = 800
    mu0 = 40.0
    sigma = 10.0
    coherence = 0.064

    # Derived values
    muA = mu0 * (1.0 + coherence)
    muB = mu0 * (1.0 - coherence)
    print("N = {}, muA = {}, muB = {}, sigma = {}".format(NA+NB, muA, muB, sigma))

    # ==============================================================
    # Create time-dependent Poisson rates for inputs to A and B
    # ==============================================================
    # Calculate number of rate bins in active time period
    activeRateBins = int(activeTime / ratePeriod + 0.5)
    # Create rates with Gaussian distribution
    gaussianA = np.random.normal(muA, sigma, activeRateBins)
    gaussianB = np.random.normal(muB, sigma, activeRateBins)
    # Calculate number of rate bins in each inactive time period,
    # assuming inactive time periods equally precede and follow
    # the active time period.
    inactiveTimeHalf = int(0.5 * (simTime - activeTime) + 0.5*second) * second
    inactiveRatePeriods = int(inactiveTimeHalf / ratePeriod + 0.5)
    ratesA = [0.0] * inactiveRatePeriods
    ratesA.extend(gaussianA)
    ratesA.extend([0.0] * inactiveRatePeriods)
    ratesB = [0.0] * inactiveRatePeriods
    ratesB.extend(gaussianB)
    ratesB.extend([0.0] * inactiveRatePeriods)
    # Time-dependencies must be defined using TimedArrays
    timedRatesA = TimedArray(ratesA * Hz, dt=ratePeriod)
    timedRatesB = TimedArray(ratesB * Hz, dt=ratePeriod)
    print("Poisson spikes generated.")
    # Show plot of rates if desired
    if showGaussian:
        LinePlot(0.0*ms, simTime, defaultclock.dt, timedRatesA, ms, Hz, "r", label = "A")
        LinePlot(0.0*ms, simTime, defaultclock.dt, timedRatesB, ms, Hz, "b", label = "B")
        yMin, yMax = plt.ylim()
        plt.ylim(bottom = min(0.0, yMin), top = max(2 * mu0, yMax))
        plt.xlabel("Time (ms)")
        plt.ylabel("Sample stimulus")
        plt.legend()
        plt.show()

    # ==============================================================
    # Create spikes used for inputs to A and B
    # ==============================================================
    # Create spikes by applying Poisson rates to Poisson generator
    PGA = PoissonGroup(NA, rates="timedRatesA(t)")
    PGB = PoissonGroup(NB, rates="timedRatesB(t)")
    MPGA = SpikeMonitor(PGA)
    MPGB = SpikeMonitor(PGB)
    netA = Network(PGA, MPGA)
    netB = Network(PGB, MPGB)
    netA.run(simTime)
    spikesA_i = MPGA.i
    spikesA_t = MPGA.t
    netB.run(simTime)
    spikesB_i = MPGB.i
    spikesB_t = MPGB.t
    # Capture spikes
    SGGA = SpikeGeneratorGroup(NA, spikesA_i, spikesA_t)
    SGGB = SpikeGeneratorGroup(NB, spikesB_i, spikesB_t)
    # Plot spikes if desired
    if showPoisson:
        plt.subplot(1, 2, 1)
        plt.plot(spikesA_i, spikesA_t / ms, "r.")
        yMin, yMax = plt.ylim()
        plt.ylim(bottom=min(0.0, yMin), top=max(simTime / ms, yMax))
        plt.title("A")
        plt.xlabel("Output index")
        plt.ylabel("Spike time (ms)")
        plt.subplot(1, 2, 2)
        plt.plot(spikesB_i, spikesB_t / ms, "b.")
        yMin, yMax = plt.ylim()
        plt.ylim(bottom=min(0.0, yMin), top=max(simTime / ms, yMax))
        plt.title("B")
        plt.xlabel("Output index")
        plt.ylabel("Spike time (ms)")
        plt.show()

    # ==============================================================
    # Define pyramidal equation and parameters
    # ==============================================================
    Cm = 0.5*nfarad
    gL = 25*nS
    vL = -70*mV
    tauR = gL/Cm
    sigmaNoise = 0.0086*volt*siemens/farad # 3 Hz mean noise
    eqPyramidal = """
    dv/dt = tauR*(vL-v) + sigmaNoise*sqrt(2*tauR)*xi/Hz : volt (unless refractory)
    """

    # ==============================================================
    # Define synapse parameters
    # ==============================================================
    w_stim = 0.8*mV
    w_rec = 0.016*mV
    w_inh = 0.016*mV

    # ==============================================================
    # Calculate noise statistics
    # ==============================================================
    # Create and run isolated network of neurons with no stimulus
    Nnoise = 50
    NGnoise = NeuronGroup(Nnoise, eqPyramidal, threshold = "v > -50*mV", reset = "v = -55*mV", refractory = 2*ms, method = "euler")
    NGnoise.v = -55*mV
    MNGnoise = SpikeMonitor(NGnoise)
    netnoise = Network(NGnoise, MNGnoise)
    netnoise.run(simTime)
    trains = MNGnoise.spike_trains()
    # Calculate period of time between spikes
    periods = []
    for train in list(trains.values()):
        if len(train) > 0:
            period = [train[0]/ms]
            i = 1
            while i < len(train):
                period.append((train[i] - train[i - 1])/ms)
                i += 1
            periods.append(period)
    # Calculate statistics
    print()
    print("=== Noise Statistics ==================================")
    print("Percent of neurons exibiting noise: {:.0f}".format(100 * len(periods) / Nnoise))
    means = []
    stddevs = []
    for period in periods:
        mean =  sum(period) / len(period)
        stddev = (sum([((x - mean) ** 2) for x in period]) / len(period)) ** 0.5
        means.append(mean)
        stddevs.append(stddev)
    if len(means) > 0:
        meanmean = sum(means) / len(means)
        meanstddev = sum(stddevs) / len(stddevs)
        stddevmean = (sum([((x - meanmean) ** 2) for x in means]) / len(means)) ** 0.5
        print("Noise: Mean of means = {:.0f} ms, StdDev of means = {:.0f} ms, Mean StdDev = {:.0f} ms".format(meanmean, stddevmean, meanstddev))
    if showNoise:
        plt.plot(MNGnoise.i, MNGnoise.t / ms, "b.")
        yMin, yMax = plt.ylim()
        plt.ylim(bottom=min(0, yMin), top=max(simTime/ms, yMax))
        plt.xlabel("Neuron index")
        plt.ylabel("Spike time (ms)")
        plt.show()

    print()
    print("=== Simulation ========================================")
    # Cortical group A
    NGA = NeuronGroup(NA, eqPyramidal, threshold = "v > -50*mV", reset = "v = -55*mV", refractory = 2*ms, method = "euler")
    NGA.v = -55*mV
    SAstim = Synapses(SGGA, NGA, on_pre = "v += w_stim")
    SAstim.connect(p=0.15)
    SArec = Synapses(NGA, NGA, on_pre = "v += w_rec")
    SArec.connect()
    MNGA = SpikeMonitor(NGA)
    NGB = NeuronGroup(NB, eqPyramidal, threshold = "v > -50*mV", reset = "v = -55*mV", refractory = 2*ms, method = "euler")
    NGB.v = -55*mV
    SBstim = Synapses(SGGB, NGB, on_pre = "v += w_stim")
    SBstim.connect(p=0.15)
    SBrec = Synapses(NGB, NGB, on_pre = "v += w_rec")
    SBrec.connect()
    MNGB = SpikeMonitor(NGB)
    SAinh = Synapses(NGB, NGA, on_pre = "v -= w_inh")
    SAinh.connect()
    SBinh = Synapses(NGA, NGB, on_pre = "v -= w_inh")
    SBinh.connect()
    netAB = Network(SGGA, NGA, SGGB, NGB, SAstim, SArec, SAinh, SBstim, SBrec, SBinh, MNGA, MNGB)
    netAB.run(simTime)

    print("Simulation finished.")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (12,5))
    fig.suptitle("Pyramidal Cells")
    ax1.plot(MNGA.i, MNGA.t / ms, "r.")
    ax2.plot(MNGB.i, MNGB.t / ms, "b.")
    ax1.set_title("A", color = "r")
    ax2.set_title("B", color = "b")
    yMin1, yMax1 = ax1.get_ylim()
    yMin2, yMax2 = ax2.get_ylim()
    ax1.set_ylim(bottom = min(0, min(yMin1, yMin2)), top = max(simTime/ms, max(yMax1, yMax2)))
    ax2.set_ylim(bottom = min(0, min(yMin1, yMin2)), top = max(simTime/ms, max(yMax1, yMax2)))
    for ax in fig.get_axes():
       ax.set(xlabel = "Neuron index", ylabel = "Spike time (ms)")
    for ax in fig.get_axes():
        ax.label_outer()
    plt.subplots_adjust(left = 0.1, right = 0.94, top = 0.86, bottom = 0.14, wspace = 0.1, hspace = 0.2)
    plt.show()
    