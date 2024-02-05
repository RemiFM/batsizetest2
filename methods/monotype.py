import numpy as np
import time
from scipy.integrate import cumtrapz
import matplotlib.pyplot as plt
import casadi as ca

"""
Calculate minimal battery size for monotype battery system (single cell technology)

    1. Calculate sizing for battery at beginning of life (BOL), make sure the battery contains enough energy and can deliver required currents
    2. Calculate cell degradation due to cycling as described in paper Mohsen (DOI: 10.3390/pr10112418), takes into account the number of cycles and depth of discharge
    3. Increase the size of the battery calculated in step 1 to offset the aging
"""

# TODO - The aging should be calculated again after step 3!


def monotype(loads, cell, V_bus, cycles=[0]):
    start_time = time.time()
    # 1. Size battery at BOL (beginning of life)
    P = loads[0]["P"].values # W
    t = loads[0]["t"].values # s
    t_soc = np.append(t, t[-1] + (t[-1] - t[-2])) # s
    M = (V_bus/cell.voltage) # Number of cells in series per string

    SOC = np.full(len(t_soc), 0.9)  # State of Charge [0-1]
    V = np.empty(len(t)) # Voltage (V)
    I = np.empty(len(t)) # Current (A)
    P_joule = np.empty(len(t)) # Joule losses (W)
    I_joule = np.empty(len(t)) # Additional current drawn due to joule losses (A)

    E_req = np.max(cumtrapz(P/1000, t/3600, initial=0))/0.8 # kWh
    N = E_req / (V_bus/cell.voltage * cell.energy)          # Number of strings in parallel

    while True:
        E = M * N * cell.energy # Energy in battery pack
        for i in range(len(t)):
            SOC[i+1] = SOC[i] - (P[i] * (t_soc[i+1]-t_soc[i]) / (E*3.6e6)) if E != 0 else SOC[i]

            V[i] = M * np.interp(SOC[i], cell.OCV_SOC, cell.OCV)
            I[i] = P[i]/V[i]

            P_joule[i] = cell.resistance*(I[i]**2)
            I_joule[i] = P_joule[i] / V[i]

        if np.max(I + I_joule) <= N*cell.dis_current: break # Check if current limits of battery pack are not exceeded, otherwise increase number of parallel strings

        N = N + 1e-2
    



    ## 2. Calculate cell degradation (aging)
    DOD = 100*(np.max(SOC) - np.min(SOC))   # Depth of Discharge [0-100]
    N_cycles = cell.aging[0]*np.exp(cell.aging[1]*DOD) + cell.aging[2]*np.exp(cell.aging[3]*DOD) # Number of cycles which can be performed before initial energy of battery is shrinked by 20%
    E_loss = E * 0.2 * (cycles[0]/N_cycles) # Energy which will be lost due to degradation at EOL (kWh)


    ## 3. Adjust sizing for aging (EOL)
    E = E + E_loss      # Battery energy (kWh) at BOL
    E_aged = E - E_loss # Battery energy (kWh) at EOL
    N = E / (M * cell.energy)

    SOC_aged = np.full(len(t_soc), 0.9) if E_aged != 0 else np.full(len(t_soc), 0)
    V_aged = np.empty(len(t))
    I_aged = np.empty(len(t))
    P_joule_aged = np.empty(len(t))
    I_joule_aged = np.empty(len(t))

    for i in range(len(t)):
        SOC[i+1]        = SOC[i]        - (P[i] * (t_soc[i+1]-t_soc[i]) / (E*3.6e6))        if E != 0       else SOC[i]
        SOC_aged[i+1]   = SOC_aged[i]   - (P[i] * (t_soc[i+1]-t_soc[i]) / (E_aged*3.6e6))   if E_aged != 0  else SOC_aged[i]

        V[i]        = M * np.interp(SOC[i], cell.OCV_SOC, cell.OCV)
        V_aged[i]   = M * np.interp(SOC_aged[i], cell.OCV_SOC, cell.OCV)

        I[i]        = P[i]/V[i]
        I_aged[i]   = P[i]/V_aged[i]

        P_joule[i]      = cell.resistance*(I[i]**2)
        P_joule_aged[i] = cell.resistance*(I_aged[i]**2)

        I_joule[i]      = P_joule[i]        / V[i]
        I_joule_aged[i] = P_joule_aged[i]   / V_aged[i]


    # Plot comparison between BOL and EOL
    # plot_comparison(t, t_soc, SOC, SOC_aged, V, V_aged, I, I_aged, I_joule, I_joule_aged, N, cell, V_bus, P_joule, P_joule_aged)

    
    # 4. Summarize results
    E_losses = np.max(cumtrapz(P_joule/1000, t/3600, initial=0)) # kWh
    E_used = np.max(cumtrapz(P/1000, t/3600, initial=0))
    efficiency = (E_used / (E_used + E_losses)) * 100

    duration = (time.time() - start_time)*1000
    print(f"Monotype solution found! \t[{duration:0.2f} ms]")

    result = {
        "t": t,
        "t_soc": t_soc,
        "P": P,
        "P_joule": P_joule,
        "SOC": SOC,
        "V": V,
        "I": I,
        "M": M,
        "N": N,
        "cost": M*N*cell.cost,
        "losses": E_losses,
        "efficiency": efficiency,
        "P_joule_aged": P_joule,
        "SOC_aged": SOC_aged,
        "V_aged": V_aged,
        "I_aged": I_aged,
        "E": E,
        "E_aged": E_aged,
        "I_rated": N * cell.dis_current,
        "time": duration
    }

    return result


def monotype2(loads, cell, V_bus, cycles=[0]):
    '''
    This functions calculates a quick initial solution, then uses CasADi to find optimal monotype solution
    '''
    start_time = time.time()
    # 1. Size battery at BOL (beginning of life)
    P = loads[0]["P"].values # W
    t = loads[0]["t"].values # s
    t_soc = np.append(t, t[-1] + (t[-1] - t[-2])) # s
    M = (V_bus/cell.voltage) # Number of cells in series per string

    SOC = np.full(len(t_soc), 0.9)  # State of Charge [0-1]
    V = np.empty(len(t)) # Voltage (V)
    I = np.empty(len(t)) # Current (A)
    P_joule = np.empty(len(t)) # Joule losses (W)
    I_joule = np.empty(len(t)) # Additional current drawn due to joule losses (A)

    E_req = np.max(cumtrapz(P/1000, t/3600, initial=0))/0.8 # kWh
    N = E_req / (V_bus/cell.voltage * cell.energy)          # Number of strings in parallel

    while True:
        E = M * N * cell.energy # Energy in battery pack
        for i in range(len(t)):
            SOC[i+1] = SOC[i] - (P[i] * (t_soc[i+1]-t_soc[i]) / (E*3.6e6)) if E != 0 else SOC[i]

            V[i] = M * np.interp(SOC[i], cell.OCV_SOC, cell.OCV)
            I[i] = P[i]/V[i]

            P_joule[i] = cell.resistance*(I[i]**2)
            I_joule[i] = P_joule[i] / V[i]

        if np.max(I + I_joule) <= N*cell.dis_current: break # Check if current limits of battery pack are not exceeded, otherwise increase number of parallel strings

        N = N + 1e-2
    
    ## 2. Calculate cell degradation (aging)
    DOD = 100*(np.max(SOC) - np.min(SOC))   # Depth of Discharge [0-100]
    N_cycles = cell.aging[0]*np.exp(cell.aging[1]*DOD) + cell.aging[2]*np.exp(cell.aging[3]*DOD) # Number of cycles which can be performed before initial energy of battery is shrinked by 20%
    E_loss = E * 0.2 * (cycles[0]/N_cycles) # Energy which will be lost due to degradation at EOL (kWh)


    ## 3. Adjust sizing for aging (EOL)
    E = E + E_loss      # Battery energy (kWh) at BOL
    E_aged = E - E_loss # Battery energy (kWh) at EOL
    N = E / (M * cell.energy)

    SOC_aged = np.full(len(t_soc), 0.9) if E_aged != 0 else np.full(len(t_soc), 0)
    V_aged = np.empty(len(t))
    I_aged = np.empty(len(t))
    P_joule_aged = np.empty(len(t))
    I_joule_aged = np.empty(len(t))

    for i in range(len(t)):
        SOC[i+1]        = SOC[i]        - (P[i] * (t_soc[i+1]-t_soc[i]) / (E*3.6e6))        if E != 0       else SOC[i]
        SOC_aged[i+1]   = SOC_aged[i]   - (P[i] * (t_soc[i+1]-t_soc[i]) / (E_aged*3.6e6))   if E_aged != 0  else SOC_aged[i]

        V[i]        = M * np.interp(SOC[i], cell.OCV_SOC, cell.OCV)
        V_aged[i]   = M * np.interp(SOC_aged[i], cell.OCV_SOC, cell.OCV)

        I[i]        = P[i]/V[i]
        I_aged[i]   = P[i]/V_aged[i]

        P_joule[i]      = cell.resistance*(I[i]**2)
        P_joule_aged[i] = cell.resistance*(I_aged[i]**2)

        I_joule[i]      = P_joule[i]        / V[i]
        I_joule_aged[i] = P_joule_aged[i]   / V_aged[i]
    
    result = {
        "t": t,
        "t_soc": t_soc,
        "P": P,
        "P_joule": P_joule,
        "SOC": SOC,
        "V": V,
        "I": I,
        "M": M,
        "N": N,
        "cost": M*N*cell.cost,
        "P_joule_aged": P_joule_aged,
        "SOC_aged": SOC_aged,
        "V_aged": V_aged,
        "I_aged": I_aged,
        "E": E,
        "E_aged": E_aged,
        "I_rated": N * cell.dis_current,
    }

    # -------------------------------------------
    opti = ca.Opti()
    #DOD = 0.8*100
    M = opti.parameter()

    # Decision variables
    P_joule = opti.variable(len(t), 1)
    P_joule_aged = opti.variable(len(t), 1)
    N = opti.variable(1, 1)
    SOC = opti.variable(len(t_soc), 1)
    SOC_aged = opti.variable(len(t_soc), 1)
    V = opti.variable(len(t), 1)
    V_aged = opti.variable(len(t), 1)
    I = opti.variable(len(t), 1)
    I_aged = opti.variable(len(t), 1)
    E_aged = opti.variable(1, 1)
    DOD = opti.variable(1, 1)
    E_used = opti.variable(len(t_soc), 1)

    # Objective function
    obj = M * N * cell.cost

    # Constraints
    opti.subject_to([
        N >= 0,
        SOC <= 0.9,
        SOC >= 0.1,
        SOC_aged <= 0.9,
        SOC_aged >= 0.1,
        P_joule == cell.resistance * (I**2),
        P_joule_aged == cell.resistance * (I_aged**2),
        E_aged == (M*N*cell.energy) * (1 - 0.2 * (cycles[0]/(cell.aging[0]*ca.exp(cell.aging[1]*DOD)+cell.aging[2]*ca.exp(cell.aging[3]*DOD)))),
        DOD == E_used[-1]/(M*N*cell.energy)*100#0.8*100#(ca.mmax(SOC) - ca.mmin(SOC))*100
    ])

    SOC[0] = ca.DM(0.9)
    E_used[0] = ca.DM(0)
    SOC_aged[0] = ca.DM(0.9)
    SOC_TO_OCV = ca.interpolant('LUT', 'bspline', [cell.OCV_SOC], cell.OCV)

    for i in range(len(t)):
        

        opti.subject_to(SOC[i+1] == ca.if_else(N == 0, SOC[i], SOC[i]-((P[i]*(t_soc[i+1]-t_soc[i]))/(M*N*cell.energy*3.6e6))))
        opti.subject_to(SOC_aged[i+1] == ca.if_else(N == 0, SOC_aged[i], SOC_aged[i]-((P[i]*(t_soc[i+1]-t_soc[i]))/(E_aged*3.6e6))))
        opti.subject_to(E_used[i+1] == ca.if_else(N==0, E_used[i], E_used[i] + P[i]*(t_soc[i+1]-t_soc[i])/3.6e6))
        opti.subject_to(V[i] == M*SOC_TO_OCV(SOC[i]))
        opti.subject_to(V_aged[i] == M*SOC_TO_OCV(SOC_aged[i]))
        opti.subject_to(I[i] == P[i]/V[i])
        opti.subject_to(I_aged[i] == P[i]/V_aged[i])
        opti.subject_to(I[i] + (P_joule[i]/V[i]) <= cell.dis_current*N)
        opti.subject_to(I_aged[i] + (P_joule_aged[i]/V_aged[i]) <= cell.dis_current*N)
        


    # Set initial values
    opti.set_value(M, V_bus/cell.voltage)
    opti.set_initial(N, result["N"])
    opti.set_initial(SOC, result["SOC"])
    opti.set_initial(SOC_aged, result["SOC_aged"])
    opti.set_initial(V, result["V"])
    opti.set_initial(V_aged, result["V_aged"])
    opti.set_initial(I, result["I"])
    opti.set_initial(I_aged, result["I_aged"])
    opti.set_initial(P_joule, result["P_joule"])
    opti.set_initial(P_joule_aged, result["P_joule_aged"])
    opti.set_initial(E_aged, result["E_aged"])
    opti.set_initial(DOD, 37)

    # Start optimization
    opti.minimize(obj)
    options = {"ipopt": {"print_level": 2, "max_iter":3000}} #level5
    opti.solver('ipopt', options)
    try:
        sol = opti.solve()
    except:
        opti.debug.show_infeasibilities()
        print("[ERROR] Optimization Failed!")
        # opti.debug.x_describe(index)
        # opti.debug.g_describe(index)
        exit()

    duration = (time.time() - start_time)*1000
    print(f"Monotype solution found! \t[{duration:0.2f} ms]")

    
    result = {
        "t": t,
        "t_soc": t_soc,
        "P": P,
        "P_joule": sol.value(P_joule),
        "SOC": sol.value(SOC),
        "V": sol.value(V),
        "I": sol.value(I),
        "M": sol.value(M),
        "N": sol.value(N),
        "cost": sol.value(M)*sol.value(N)*cell.cost,
        "P_joule_aged": sol.value(P_joule_aged),
        "SOC_aged": sol.value(SOC_aged),
        "V_aged": sol.value(V_aged),
        "I_aged": sol.value(I_aged),
        "E": sol.value(M)*sol.value(N)*cell.energy,
        "E_aged": sol.value(E_aged),
        "I_rated": sol.value(N) * cell.dis_current,
        "time": duration
    }

    print(f"--> Depth of Discharge: {sol.value(DOD)}")


    return result





def monotype_multi(loads, cell, V_bus, cycles=[0]):
    '''
    This functions calculates a quick initial solution, then uses CasADi to find optimal monotype solution
    '''
    start_time = time.time()

    M = (V_bus/cell.voltage)
    N, P, t, t_soc, SOC, V, I, P_joule, I_joule, E_aged, SOC_aged, V_aged, I_aged, P_joule_aged, I_joule_aged, DOD = [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], []

    for i in range(len(loads)):
        P.append(loads[i]["P"].values) # (W)
        t.append(loads[i]["t"].values) # (s)
        t_soc.append(np.append(t[i], t[i][-1] + (t[i][-1] - t[i][-2]))) # (s)

        SOC.append(np.full(len(t_soc[i]), 0.9))  # State of Charge [0-1]
        V.append(np.empty(len(t[i])))            # Voltage (V)
        I.append(np.empty(len(t[i])))            # Current (A)
        P_joule.append(np.empty(len(t[i])))      # Joule losses (W)
        I_joule.append(np.empty(len(t[i])))      # Additional current drawn due to joule losses (A)

        E_req = np.max(cumtrapz(P[i]/1000, t[i]/3600, initial=0))/0.8     # Required energy (kWh)
        N.append(E_req / (V_bus/cell.voltage * cell.energy))        # Number of strings in parallel

        while True:
            E = M * N[i] * cell.energy # Energy in battery pack
            for j in range(len(t[i])):
                SOC[i][j+1] = SOC[i][j] - (P[i][j] * (t_soc[i][j+1]-t_soc[i][j]) / (E*3.6e6)) if E != 0 else SOC[i][j]

                V[i][j] = M * np.interp(SOC[i][j], cell.OCV_SOC, cell.OCV)
                I[i][j] = P[i][j]/V[i][j]

                P_joule[i][j] = cell.resistance*(I[i][j]**2)
                I_joule[i][j] = P_joule[i][j] / V[i][j]

            if np.max(I[i] + I_joule[i]) <= N[i]*cell.dis_current: break # Check if current limits of battery pack are not exceeded, otherwise increase number of parallel strings

            N[i] += 5e-2

        # ## 2. Calculate cell degradation (aging)
        DOD.append(100*(np.max(SOC[i]) - np.min(SOC[i])))   # Depth of Discharge [0-100]
        N_cycles = cell.aging[0]*np.exp(cell.aging[1]*DOD[i]) + cell.aging[2]*np.exp(cell.aging[3]*DOD[i]) # Number of cycles which can be performed before initial energy of battery is shrinked by 20%
        E_loss = E * 0.2 * (cycles[i]/N_cycles) # Energy which will be lost due to degradation at EOL (kWh)

        ## 3. Adjust sizing for aging (EOL)
        E = E + E_loss      # Battery energy (kWh) at BOL
        E_aged.append(E - E_loss) # Battery energy (kWh) at EOL

        N[i] = E / (M * cell.energy)

        # ----------------------------------------------------------

    for i in range(len(loads)):
        E = max(N) * M * cell.energy

        for j in range(len(t[i])):
            SOC[i][j+1] = SOC[i][j] - (P[i][j] * (t_soc[i][j+1]-t_soc[i][j]) / (E*3.6e6)) if E != 0 else SOC[i][j]
            V[i][j] = M * np.interp(SOC[i][j], cell.OCV_SOC, cell.OCV)
            I[i][j] = P[i][j]/V[i][j]
            P_joule[i][j] = cell.resistance*(I[i][j]**2)
            I_joule[i][j] = P_joule[i][j] / V[i][j]

        DOD[i] = 100*(np.max(SOC[i]) - np.min(SOC[i]))   # Depth of Discharge [0-100]
        N_cycles = cell.aging[0]*np.exp(cell.aging[1]*DOD[i]) + cell.aging[2]*np.exp(cell.aging[3]*DOD[i]) # Number of cycles which can be performed before initial energy of battery is shrinked by 20%
        E_loss = E * 0.2 * (cycles[i]/N_cycles) # Energy which will be lost due to degradation at EOL (kWh)
        E_aged[i] = (E - E_loss) # Battery energy (kWh) at EOL

        SOC_aged.append(np.full(len(t_soc[i]), 0.9))
        V_aged.append(np.empty(len(t[i])))
        I_aged.append(np.empty(len(t[i])))
        P_joule_aged.append(np.empty(len(t[i])))
        I_joule_aged.append(np.empty(len(t[i])))

        for j in range(len(t[i])):
            SOC_aged[i][j+1] = SOC_aged[i][j] - (P[i][j] * (t_soc[i][j+1]-t_soc[i][j]) / (E_aged[i]*3.6e6)) if E_aged[i] != 0  else SOC_aged[i][j]           
            V_aged[i][j] = M * np.interp(SOC_aged[i][j], cell.OCV_SOC, cell.OCV)           
            I_aged[i][j] = P[i][j]/V_aged[i][j]           
            P_joule_aged[i][j] = cell.resistance*(I_aged[i][j]**2)           
            I_joule_aged[i][j] = P_joule_aged[i][j]   / V_aged[i][j]

    duration = (time.time() - start_time)*1000
    
    result = {
        "t": t,
        "t_soc": t_soc,
        "P": P,
        "P_joule": P_joule,
        "SOC": SOC,
        "V": V,
        "I": I,
        "M": M,
        "N": max(N),
        "DOD": DOD,
        "cost": M*max(N)*cell.cost,
        "P_joule_aged": P_joule_aged,
        "SOC_aged": SOC_aged,
        "V_aged": V_aged,
        "I_aged": I_aged,
        "E": M*max(N)*cell.energy,
        "E_aged": E_aged,
        "I_rated": max(N) * cell.dis_current,
        "time": duration
    }


    ## CASADI OPTIMIZATION
    opti = ca.Opti()
    M = opti.parameter()

    # Decision variables
    N = opti.variable(1, 1)
    DOD = opti.variable(len(loads), 1)
    E_aged = opti.variable(len(loads), 1)

    SOC1 = opti.variable(len(t_soc[0]), 1)
    V1 = opti.variable(len(t[0]), 1)
    I1 = opti.variable(len(t[0]), 1)
    P_joule1 = opti.variable(len(t[0]), 1)
    SOC1_aged = opti.variable(len(t_soc[0]), 1)
    V1_aged = opti.variable(len(t[0]), 1)
    I1_aged = opti.variable(len(t[0]), 1)
    P_joule1_aged = opti.variable(len(t[0]), 1)
    E_used1 = opti.variable(len(t_soc[0]), 1)

    if len(loads) >= 2:
        SOC2 = opti.variable(len(t_soc[1]), 1)
        V2 = opti.variable(len(t[1]), 1)
        I2 = opti.variable(len(t[1]), 1)
        P_joule2 = opti.variable(len(t[1]), 1)
        SOC2_aged = opti.variable(len(t_soc[1]), 1)
        V2_aged = opti.variable(len(t[1]), 1)
        I2_aged = opti.variable(len(t[1]), 1)
        P_joule2_aged = opti.variable(len(t[1]), 1)
        E_used2 = opti.variable(len(t_soc[1]), 1)

    # Objective function
    obj = M * N * cell.cost

    # Constraints
    opti.subject_to([
        N >= 0,
        SOC1 <= 0.9,
        SOC1 >= 0.1,
        SOC1_aged <= 0.9,
        SOC1_aged >= 0.1,
        P_joule1 == cell.resistance * (I1**2),
        P_joule1_aged == cell.resistance * (I1_aged**2),
        
        E_aged[0] == (M*N*cell.energy) * (1 - 0.2 * (cycles[0]/(cell.aging[0]*ca.exp(cell.aging[1]*DOD[0])+cell.aging[2]*ca.exp(cell.aging[3]*DOD[0])))),
        DOD[0] == E_used1[-1]/(M*N*cell.energy)*100#0.8*100#(ca.mmax(SOC) - ca.mmin(SOC))*100
    ])

    SOC1[0] = ca.DM(0.9)
    E_used1[0] = ca.DM(0)
    SOC1_aged[0] = ca.DM(0.9)

    if len(loads) >= 2:
        opti.subject_to([
            SOC2 <= 0.9,
            SOC2 >= 0.1,
            SOC2_aged <= 0.9,
            SOC2_aged >= 0.1,
            P_joule2 == cell.resistance * (I2**2),
            P_joule2_aged == cell.resistance * (I2_aged**2),
            
            E_aged[1] == (M*N*cell.energy) * (1 - 0.2 * (cycles[1]/(cell.aging[0]*ca.exp(cell.aging[1]*DOD[1])+cell.aging[2]*ca.exp(cell.aging[3]*DOD[1])))),
            DOD[1] == E_used2[-1]/(M*N*cell.energy)*100#0.8*100#(ca.mmax(SOC) - ca.mmin(SOC))*100
        ])

        SOC2[0] = ca.DM(0.9)
        E_used2[0] = ca.DM(0)
        SOC2_aged[0] = ca.DM(0.9)
       
    SOC_TO_OCV = ca.interpolant('LUT', 'bspline', [cell.OCV_SOC], cell.OCV)

    for i in range(len(t[0])):
        opti.subject_to(SOC1[i+1] == ca.if_else(N == 0, SOC1[i], SOC1[i]-((P[0][i]*(t_soc[0][i+1]-t_soc[0][i]))/(M*N*cell.energy*3.6e6))))
        opti.subject_to(SOC1_aged[i+1] == ca.if_else(N == 0, SOC1_aged[i], SOC1_aged[i]-((P[0][i]*(t_soc[0][i+1]-t_soc[0][i]))/(E_aged[0]*3.6e6))))
        opti.subject_to(E_used1[i+1] == ca.if_else(N==0, E_used1[i], E_used1[i] + P[0][i]*(t_soc[0][i+1]-t_soc[0][i])/3.6e6))
        opti.subject_to(V1[i] == M*SOC_TO_OCV(SOC1[i]))
        opti.subject_to(V1_aged[i] == M*SOC_TO_OCV(SOC1_aged[i]))
        opti.subject_to(I1[i] == P[0][i]/V1[i])
        opti.subject_to(I1_aged[i] == P[0][i]/V1_aged[i])
        opti.subject_to(I1[i] + (P_joule1[i]/V1[i]) <= cell.dis_current*N)
        opti.subject_to(I1_aged[i] + (P_joule1_aged[i]/V1_aged[i]) <= cell.dis_current*N)

    if len(loads) >= 2:
        for i in range(len(t[1])):
            opti.subject_to(SOC2[i+1] == ca.if_else(N == 0, SOC2[i], SOC2[i]-((P[1][i]*(t_soc[1][i+1]-t_soc[1][i]))/(M*N*cell.energy*3.6e6))))
            opti.subject_to(SOC2_aged[i+1] == ca.if_else(N == 0, SOC2_aged[i], SOC2_aged[i]-((P[1][i]*(t_soc[1][i+1]-t_soc[1][i]))/(E_aged[1]*3.6e6))))
            opti.subject_to(E_used2[i+1] == ca.if_else(N==0, E_used2[i], E_used2[i] + P[1][i]*(t_soc[1][i+1]-t_soc[1][i])/3.6e6))
            opti.subject_to(V2[i] == M*SOC_TO_OCV(SOC2[i]))
            opti.subject_to(V2_aged[i] == M*SOC_TO_OCV(SOC2_aged[i]))
            opti.subject_to(I2[i] == P[1][i]/V2[i])
            opti.subject_to(I2_aged[i] == P[1][i]/V2_aged[i])
            opti.subject_to(I2[i] + (P_joule2[i]/V2[i]) <= cell.dis_current*N)
            opti.subject_to(I2_aged[i] + (P_joule2_aged[i]/V2_aged[i]) <= cell.dis_current*N)

    
    # Set initial values
    opti.set_value(M, V_bus/cell.voltage)
    opti.set_initial(N, result["N"])
    opti.set_initial(SOC1, result["SOC"][0])
    opti.set_initial(SOC1_aged, result["SOC_aged"][0])
    opti.set_initial(V1, result["V"][0])
    opti.set_initial(V1_aged, result["V_aged"][0])
    opti.set_initial(I1, result["I"][0])
    opti.set_initial(I1_aged, result["I_aged"][0])
    opti.set_initial(P_joule1, result["P_joule"][0])
    opti.set_initial(P_joule1_aged, result["P_joule_aged"][0])
    opti.set_initial(E_aged, result["E_aged"])
    opti.set_initial(DOD, result["DOD"])

    if len(loads) >= 2:
        opti.set_initial(SOC2, result["SOC"][1])
        opti.set_initial(SOC2_aged, result["SOC_aged"][1])
        opti.set_initial(V2, result["V"][1])
        opti.set_initial(V2_aged, result["V_aged"][1])
        opti.set_initial(I2, result["I"][1])
        opti.set_initial(I2_aged, result["I_aged"][1])
        opti.set_initial(P_joule2, result["P_joule"][1])
        opti.set_initial(P_joule2_aged, result["P_joule_aged"][1])

    # Start optimization
    opti.minimize(obj)
    options = {"ipopt": {"print_level": 2, "max_iter":3000}} #level5
    opti.solver('ipopt', options)
    try:
        sol = opti.solve()
    except:
        opti.debug.show_infeasibilities()
        print("[ERROR] Optimization Failed!")
        exit()

    duration = (time.time() - start_time)*1000

    result = {
        "t": t,
        "t_soc": t_soc,
        "P": P,
        "P_joule": [sol.value(P_joule1)] if len(loads) == 1 else [sol.value(P_joule1), sol.value(P_joule2)],
        "SOC": [sol.value(SOC1)] if len(loads) == 1 else [sol.value(SOC1), sol.value(SOC2)],
        "V": [sol.value(V1)] if len(loads) == 1 else [sol.value(V1), sol.value(V2)],
        "I": [sol.value(I1)] if len(loads) == 1 else [sol.value(I1), sol.value(I2)],
        "M": sol.value(M),
        "N": sol.value(N),
        "DOD": sol.value(DOD),
        "cost": sol.value(M)*sol.value(N)*cell.cost,
        "P_joule_aged": [sol.value(P_joule1_aged)] if len(loads) == 1 else [sol.value(P_joule1_aged), sol.value(P_joule2_aged)],
        "SOC_aged": [sol.value(SOC1_aged)] if len(loads) == 1 else [sol.value(SOC1_aged), sol.value(SOC2_aged)],
        "V_aged": [sol.value(V1_aged)] if len(loads) == 1 else [sol.value(V1_aged), sol.value(V2_aged)],
        "I_aged": [sol.value(I1_aged)] if len(loads) == 1 else [sol.value(I1_aged), sol.value(I2_aged)],
        "E": sol.value(M)*sol.value(N)*cell.energy,
        "E_aged": sol.value(E_aged),
        "I_rated": sol.value(N) * cell.dis_current,
        "time": duration
    }

    return result








def plot_comparison(t, t_soc, SOC, SOC_aged, V, V_aged, I, I_aged, I_joule, I_joule_aged, N, cell, V_bus, P_joule, P_joule_aged):
    plt.subplot(2, 2, 1)
    plt.plot(t_soc/3600, 100 * SOC, label="intitial")
    plt.plot(t_soc/3600, 100 * SOC_aged, label="aged")
    plt.title("State of Charge Evolution")
    plt.xlabel("Time (h)")  # Add x-axis label
    plt.ylabel("State of Charge (%)")  # Add y-axis label
    plt.ylim(0, 100)
    plt.axhline(y=10, color=plt.rcParams['grid.color'], alpha=plt.rcParams['grid.alpha'], linestyle='--') 
    plt.axhline(y=90, color=plt.rcParams['grid.color'], alpha=plt.rcParams['grid.alpha'], linestyle='--')
    plt.grid(True)
    plt.legend()


    plt.subplot(2, 2, 3)
    plt.plot(t/3600, V, label="intitial")
    plt.plot(t/3600, V_aged, label="aged")
    plt.title("Voltage Evolution")
    plt.xlabel("Time (h)")  # Add x-axis label
    plt.ylabel("Voltage (V)")  # Add y-axis label
    plt.axhline(y=V_bus, color=plt.rcParams['grid.color'], alpha=plt.rcParams['grid.alpha'], linestyle='--') 
    plt.grid(True)
    plt.legend()

    plt.subplot(2, 2, 2)
    plt.plot(t/3600, I + I_joule, label="intitial")
    plt.plot(t/3600, I_aged + I_joule_aged, label="aged")
    plt.title("Current Evolution")
    plt.xlabel("Time (h)")  # Add x-axis label
    plt.ylabel("Current (A)")  # Add y-axis label
    plt.axhline(y=N*cell.dis_current, color=plt.rcParams['grid.color'], alpha=plt.rcParams['grid.alpha'], linestyle='--', label=f"limit: {N*cell.dis_current:0.2f}") 
    plt.grid(True)
    plt.legend()

    plt.subplot(2, 2, 4)
    plt.plot(t/3600, P_joule/1000, label="intitial")
    plt.plot(t/3600, P_joule_aged/1000, label="aged")
    plt.title("Joule losses")
    plt.xlabel("Time (h)")  # Add x-axis label
    plt.ylabel("Power (kW)")  # Add y-axis label
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.show()

    return None