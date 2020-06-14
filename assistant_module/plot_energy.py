import numpy as np
from assistant_module.get_energy import get_energy


def plot_energy(n, ax, switch='conventional', structure='conventional', marker='v'):
    """
    plot the energy consumption of all possible decision level before the last comparision.
    :param n: number of bits
    :param ax: Axes of the plot
    :param switch: switching method, 'conventional' or 'monotonic'
    :param structure: structure of ADC
    :param marker: marker of the curve
    :return: a plot of energy consumption
    """
    # possible decision level before the last comparision
    code_decimal = np.arange(1, 2 ** n, 2)
    sw_energy_sum = get_energy(n, switch=switch, structure=structure)
    ax.plot(code_decimal, sw_energy_sum, marker=marker, label=switch, markevery=0.05)
    # axis.grid()
    ax.set_xlabel('Output Code')
    ax.set_ylabel(r'Switching Energy ($C_0V_{ref}^2$)')
    ax.set_title('Switching Energy Consumption')
