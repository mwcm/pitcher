import numpy as np
from assistant_module.get_decision_path import get_decision_path


def get_energy(n, switch='conventional', structure='conventional'):
    """
    get the energy consumption of every code, each code represents the possible decision level before the last
    decision(a odd decimal integer).
    :param n: resolution of DAC
    :param switch: switching method: 'conventional': conventional one-step switching
                                     'monotonic': monotonic capacitor switching, in each transition step, only one
                                                  capacitor in one side is switched.
                                     'mcs': merged capacitor switching
                                     'split': split-capacitor method. The MSB capacitor is split into a copy of the
                                            rest of the capacitor array. When down-switching occurs, only the
                                            corresponding capacitor in the sub-capacitor array is discharged to the
                                            ground
    :param structure: structure of ADC: 'conventional': conventional single-ended structure
                                        'differential': has two arrays of capacitors, the switch states of positive and
                                                        negative side are complementary. The energy consumption is two
                                                        times of that in the conventional structure, if conventional
                                                        switching method is used.
    :return: a ndarray, each element represents the energy consumption of each code.
    """
    # possible decision level before the last comparision
    code_decimal = np.arange(1, 2 ** n, 2)
    decision_path = get_decision_path(n)  # two-dimensional
    # store the switching energy of each code
    sw_energy_sum = np.zeros(len(code_decimal))
    if switch == 'conventional':
        coefficient = 1
        if structure == 'differential':
            # the switching states of both sides are complementary, so that the energy consumption is two times of
            # that in conventional(single-ended) structure.
            coefficient = 2
        for i in range(len(code_decimal)):
            # weight of each decision threshold layer
            weights_ideal = [0.5 ** (i + 1) for i in range(n)]
            sw_energy = np.zeros(n)
            sw_energy[0] = 0.5 * decision_path[i, 0]

            # calculate the energy for up-switching steps
            # 1 is the index offset
            sw_up_pos = np.where(
                decision_path[i, 1:] > decision_path[i, 0:-1])[0] + 1
            # print(code_decimal[i],' sw_up_pos: ',sw_up_pos)
            if not sw_up_pos.size == 0:
                # sw_energy[sw_up_pos] = decision_path[i,sw_up_pos]*(-1)*(weights_ideal[sw_up_pos])+ 2**(n-1-sw_up_pos)
                # 2**(n-1-sw_up_pos) stands for E_sw = C_up*V_ref^2
                for k in sw_up_pos:
                    # \delta V_x is positive,so *(-1)
                    sw_energy[k] = decision_path[i, k] * \
                        (-1) * (weights_ideal[k]) + 2**(n - 1 - k)

            sw_dn_pos = np.where(
                decision_path[i, 1:] < decision_path[i, 0:-1])[0] + 1
            # print(code_decimal[i],' sw_dn_pos: ',sw_dn_pos)
            if not sw_dn_pos.size == 0:
                # sw_energy[sw_dn_pos] = decision_path[i,sw_dn_pos]*(-1)*(weights_ideal[sw_dn_pos]) + 2**(n-1-sw_dn_pos)
                for k in sw_dn_pos:
                    sw_energy[k] = decision_path[i, k] * \
                        (weights_ideal[k]) + 2**(n - 1 - k)
            # print(code_decimal[i],': ',sw_energy)
            sw_energy_sum[i] = np.sum(sw_energy)
        return coefficient * sw_energy_sum

    if switch == 'monotonic':
        if structure == 'conventional':
            raise Exception(
                'Conventional(single-ended) structure does not support monotonic switching.')
        for i in range(len(code_decimal)):
            # the total capacitance of positive and negative sides
            c_tp = c_tn = 2 ** (n - 1)
            # vx unchanged in the first step
            weights_ideal = np.concatenate(
                ([0], [0.5 ** j for j in range(1, n)]))
            sw_energy = np.zeros(n)
            sw_energy[0] = 0

            # define an array to store the switching types(up or down) of each
            # step.
            sw_process = np.zeros(n)
            # find the up-switching and down-switching steps
            # 1 is the index offset
            sw_up_pos = np.where(
                decision_path[i, 1:] > decision_path[i, 0:-1])[0] + 1
            sw_dn_pos = np.where(
                decision_path[i, 1:] < decision_path[i, 0:-1])[0] + 1
            sw_process[sw_up_pos], sw_process[sw_dn_pos] = 1, 0
            for k in range(1, n):
                # if up-switching occurs, a capacitor of the p-side will be connected to the ground while n-side remains
                # unchanged; if down-switching occurs, a capacitor of n -side will be connected to the ground while
                # p-side remains unchanged. Attention: here is the range(1,n), when k starts from 1, the first
                # capacitor switched to the ground is 2**(n-2)*C0 ( the MSB capacitor differs from which in the
                # conventional case.
                c_tp = c_tp - 2**(n - 1 - k) * sw_process[k]
                c_tn = c_tn - 2**(n - 1 - k) * (1 - sw_process[k])
                sw_energy[k] = c_tp * (-1) * (- weights_ideal[k]) * sw_process[k] + \
                    c_tn * (-1) * (- weights_ideal[k]) * (1 - sw_process[k])
            sw_energy_sum[i] = np.sum(sw_energy)
        return sw_energy_sum

    if switch == 'mcs':
        if structure == 'conventional':
            raise Exception(
                'Conventional(single-ended) structure does not support monotonic switching.')
        weights_ideal = np.concatenate(
            ([0.5 ** j for j in range(1, n)], [0.5 ** (n - 1)]))
        cap_ideal = np.concatenate(
            ([2 ** (n - 2 - j) for j in range(n - 1)], [1]))
        for i in range(len(code_decimal)):
            sw_energy = np.zeros(n)

            # find the up-switching and down-switching steps
            # 1 is the index offset
            sw_up_pos = np.where(
                decision_path[i, 1:] > decision_path[i, 0:-1])[0] + 1
            sw_dn_pos = np.where(
                decision_path[i, 1:] < decision_path[i, 0:-1])[0] + 1
            # connection of bottom plates of positive and negative capacitor arrays.
            # at the sampling phase, all the bottom plates are connected to Vcm
            # = 0.5* Vref
            cap_connect_p = np.full((n, n), 0.5)
            cap_connect_n = np.full((n, n), 0.5)
            # define an array to store the switching types(up or down) of each
            # step.
            sw_process = np.zeros(n)
            sw_process[sw_up_pos], sw_process[sw_dn_pos] = 1.0, 0
            # store the v_x of both sides in each step, here the term v_ip and
            # v_in are subtracted.
            v_xp = np.zeros(n)
            v_xn = np.zeros(n)
            # store the voltage difference between the plates of each capacitor in each step, here the term v_ip- v_cm
            # and v_in - v_cm are subtracted, because when calculating the change of v_cap, these terms are constant and
            # so eliminated.
            v_cap_p = np.zeros((n, n))
            v_cap_n = np.zeros((n, n))

            for k in range(1, n):
                # update the connections of bottom plates
                cap_connect_p[k:, k - 1], cap_connect_n[k:,
                                                        k - 1] = 1 - sw_process[k], sw_process[k]

                v_xp[k] = np.inner(cap_connect_p[k], weights_ideal)
                v_xn[k] = np.inner(cap_connect_n[k], weights_ideal)
                # calculate the voltage across the top and bottom plates of
                # capacitors
                v_cap_p[k] = v_xp[k] - cap_connect_p[k]
                v_cap_n[k] = v_xn[k] - cap_connect_n[k]
                # find index of  the capacitors connected to the reference
                # voltage
                c_tp_index = np.where(cap_connect_p[k] == 1.0)[0]
                c_tn_index = np.where(cap_connect_n[k] == 1.0)[0]
                # energy = - V_ref * ∑(c_t[j] * ∆v_cap[j])
                sw_energy_p = - \
                    np.inner(cap_ideal[c_tp_index], (v_cap_p[k, c_tp_index] - v_cap_p[k - 1, c_tp_index]))
                sw_energy_n = - \
                    np.inner(cap_ideal[c_tn_index], (v_cap_n[k, c_tn_index] - v_cap_n[k - 1, c_tn_index]))
                sw_energy[k] = sw_energy_p + sw_energy_n
            sw_energy_sum[i] = np.sum(sw_energy)
        return sw_energy_sum

    if switch == 'split':
        coefficient = 1
        if structure == 'differential':
            coefficient = 2
        if n < 2:
            raise Exception(
                "Number of bits must be greater than or equal to 2. ")
        # capacitor array, cap_ideal has the shape of (2,n), in which the first row is the sub-capacitor array of the
        # MSB capacitor, the second row is the main capacitor array(excluding
        # the MSB capacitor)
        cap_ideal = np.repeat(np.concatenate(
            ([2**(n - 2 - i) for i in range(n - 1)], [1]))[np.newaxis, :], 2, axis=0)
        weights_ideal = cap_ideal / (2**n)
        for i in range(len(code_decimal)):
            sw_energy = np.zeros(n)
            sw_energy[0] = 0.5 * decision_path[i, 0]
            # find the up-switching and down-switching steps
            # 1 is the index offset
            sw_up_pos = np.where(
                decision_path[i, 1:] > decision_path[i, 0:-1])[0] + 1
            sw_dn_pos = np.where(
                decision_path[i, 1:] < decision_path[i, 0:-1])[0] + 1
            # define an array to store the switching types(up or down) of each
            # step.
            sw_process = np.zeros(n)
            sw_process[sw_up_pos], sw_process[sw_dn_pos] = 1.0, 0
            # store the bottom plates connection in each step
            cap_connect = np.repeat(
                np.vstack(
                    (np.ones(n), np.zeros(n)))[
                    np.newaxis, :, :], n, axis=0)
            # store the voltage at X point ,here the term v_cm - v_in is
            # subtracted
            v_x = np.zeros(n)
            v_x[0] = np.sum(np.multiply(weights_ideal, cap_connect[0]))
            # the voltage between top plates and bottom plates
            v_cap = np.zeros((n, 2, n))
            v_cap[0] = v_x[0] - cap_connect[0]
            for k in range(1, n):
                # if up-switching: the capacitor with index k-1 in the main capacitor array will be charged to V_ref,
                # and the capacitor with same index remains charged to V_ref; if down-switching: the capacitor
                # with index k-1 in the sub-capacitor array will be discharged to ground, and the capacitor with the
                # same index remains discharged.
                cap_connect[k:, :, k - 1] = sw_process[k]
                v_x[k] = np.sum(np.multiply(weights_ideal, cap_connect[k]))
                v_cap[k] = v_x[k] - cap_connect[k]
                # find index of  the capacitors charged to the reference
                # voltage
                c_t_index = np.where(
                    cap_connect[k] == 1.0)  # 2-dimensional index
                # energy = - V_ref * ∑(c_t[j] * ∆v_cap[j])
                # attention that v_cap is 3d-array, the the slicing index
                # should also be 3-dimensional
                sw_energy[k] = - np.inner(cap_ideal[c_t_index],
                                          (v_cap[k,c_t_index[0],
                                                 c_t_index[-1]] - v_cap[k - 1, c_t_index[0], c_t_index[-1]]))
            sw_energy_sum[i] = np.sum(sw_energy)
        return coefficient * sw_energy_sum
