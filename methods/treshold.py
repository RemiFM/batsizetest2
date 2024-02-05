import numpy as np
from scipy.integrate import cumtrapz
import time
import casadi as ca

def treshold(loads, cell_HE, cell_HP, V_bus, cycles):
    start_time = time.time()

    P = loads[0]["P"].values # W
    t = loads[0]["t"].values # s
    t_soc = np.append(t, t[-1] + (t[-1] - t[-2])) # s

    ## -- Rule Based Initial Solution
    P_HE = np.empty(len(t))
    P_HP = np.empty(len(t))
    P_HE_joule = np.empty(len(t))
    P_HP_joule = np.empty(len(t))
    I_HE_joule = np.empty(len(t))
    I_HP_joule = np.empty(len(t))
    SOC_HE = np.full(len(t_soc), 0.9)
    SOC_HP = np.full(len(t_soc), 0.9)
    V_HE = np.empty(len(t))
    V_HP = np.empty(len(t))
    I_HE = np.empty(len(t))
    I_HP = np.empty(len(t))

    #cost = np.empty(int(np.max(P)/1000))
    resolution = 1000*3*2*2*2
    cost = []
    N_HE_list = []
    N_HP_list = []
    limit = []

    M_HE = V_bus/cell_HE.voltage
    M_HP = V_bus/cell_HP.voltage

    for P_limit in range(0, round(np.max(P))+resolution, resolution):
        
        for i in range(len(t)):
            if P[i] > P_limit:
                P_HE[i] = P_limit
                P_HP[i] = P[i] - P_limit
            else:
                P_HE[i] = P[i]
                P_HP[i] = 0

        E_HE = cumtrapz(P_HE/1000, t/3600, initial=0) # kWh
        E_HP = cumtrapz(P_HP/1000, t/3600, initial=0) # kWh

        E_HE_req1 = np.max(E_HE)/0.8 # kWh
        E_HP_req1 = np.max(E_HP)/0.8 # kWh
        N_HE = E_HE_req1 / (M_HE * cell_HE.energy) 
        N_HP = E_HP_req1 / (M_HP * cell_HP.energy) 

        while True:
            E_HE = M_HE * N_HE * cell_HE.energy
            E_HP = M_HP * N_HP * cell_HP.energy

            for i in range(len(t)):
                SOC_HE[i+1] = SOC_HE[i] - (P_HE[i] * (t_soc[i+1]-t_soc[i]) / (E_HE*3.6e6)) if E_HE_req1 != 0 else SOC_HE[i]
                SOC_HP[i+1] = SOC_HP[i] - (P_HP[i] * (t_soc[i+1]-t_soc[i]) / (E_HP*3.6e6)) if E_HP_req1 != 0 else SOC_HP[i]

                V_HE[i] = V_bus/cell_HE.voltage * np.interp(SOC_HE[i], cell_HE.OCV_SOC, cell_HE.OCV)
                V_HP[i] = V_bus/cell_HP.voltage * np.interp(SOC_HP[i], cell_HP.OCV_SOC, cell_HP.OCV)

                I_HE[i] = P_HE[i]/V_HE[i]
                I_HP[i] = P_HP[i]/V_HP[i]

                P_HE_joule[i] = cell_HE.resistance * (I_HE[i]**2)
                P_HP_joule[i] = cell_HP.resistance * (I_HP[i]**2)

                I_HE_joule[i] = P_HE_joule[i] / V_HE[i]
                I_HP_joule[i] = P_HP_joule[i] / V_HP[i]
            
            if np.max(I_HE + I_HE_joule) > N_HE*cell_HE.dis_current: N_HE += 1e-0
            if np.max(I_HP + I_HP_joule) > N_HP*cell_HP.dis_current: N_HP += 1e-0
            if np.max(I_HE + I_HE_joule) <= N_HE*cell_HE.dis_current and np.max(I_HP + I_HP_joule) <= N_HP*cell_HP.dis_current: break


        # N_HE_req2 = (np.max(I_HE) + np.max(I_HE_joule)) / cell_HE.dis_current
        # N_HP_req2 = (np.max(I_HP) + np.max(I_HP_joule)) / cell_HP.dis_current

        cost.append(V_bus/cell_HE.voltage * N_HE * cell_HE.cost +
                    V_bus/cell_HP.voltage * N_HP * cell_HP.cost)
        
        N_HE_list.append(N_HE)
        N_HP_list.append(N_HP)
        limit.append(P_limit)

        progress = (P_limit/(np.max(P)+resolution)) * 100
        print(f"Calculating rule-based...\t[{int(progress/4)*'='}{int((100-progress)/4)*' '}] \t{progress:.2f}%", end="\r", flush=True)



    # Find least expensive solution
    index_min = cost.index(min(cost))
    #print(f"Treshold Limit: {limit[index_min]/1000:0.2f} kW")

    P_limit = limit[index_min]
    for i in range(len(t)):
        if P[i] > P_limit:
            P_HE[i] = P_limit
            P_HP[i] = P[i] - P_limit
        else:
            P_HE[i] = P[i]
            P_HP[i] = 0

    E_HE = M_HE * N_HE_list[index_min] * cell_HE.energy
    E_HP = M_HP * N_HP_list[index_min] * cell_HP.energy

    for i in range(len(t)):
        SOC_HE[i+1] = SOC_HE[i] - (P_HE[i] * (t_soc[i+1]-t_soc[i]) / (E_HE*3.6e6)) if E_HE != 0 else SOC_HE[i]
        SOC_HP[i+1] = SOC_HP[i] - (P_HP[i] * (t_soc[i+1]-t_soc[i]) / (E_HP*3.6e6)) if E_HP != 0 else SOC_HP[i]

        V_HE[i] = V_bus/cell_HE.voltage * np.interp(SOC_HE[i], cell_HE.OCV_SOC, cell_HE.OCV)
        V_HP[i] = V_bus/cell_HP.voltage * np.interp(SOC_HP[i], cell_HP.OCV_SOC, cell_HP.OCV)

        I_HE[i] = P_HE[i]/V_HE[i]
        I_HP[i] = P_HP[i]/V_HP[i]

        P_HE_joule[i] = cell_HE.resistance * I_HE[i]**2
        P_HP_joule[i] = cell_HP.resistance * I_HP[i]**2



    # Calculate cell degradation (aging)
    DOD_HE = 100*(np.max(SOC_HE) - np.min(SOC_HE))   # Depth of Discharge [0-100]
    DOD_HP = 100*(np.max(SOC_HP) - np.min(SOC_HP))   # Depth of Discharge [0-100]
    N_cycles_HE = cell_HE.aging[0]*np.exp(cell_HE.aging[1]*DOD_HE) + cell_HE.aging[2]*np.exp(cell_HE.aging[3]*DOD_HE) # Number of cycles which can be performed before initial energy of HE battery is shrinked by 20%
    N_cycles_HP = cell_HP.aging[0]*np.exp(cell_HP.aging[1]*DOD_HP) + cell_HP.aging[2]*np.exp(cell_HP.aging[3]*DOD_HP) # Number of cycles which can be performed before initial energy of HP battery is shrinked by 20%
    
    E_HE_loss = E_HE * 0.2 * (cycles[0]/N_cycles_HE) # Energy which will be lost due to degradation at EOL (kWh)
    E_HP_loss = E_HP * 0.2 * (cycles[0]/N_cycles_HP) # Energy which will be lost due to degradation at EOL (kWh)


    # Adjust sizing for aging (EOL)
    E_HE = E_HE + E_HE_loss
    E_HP = E_HP + E_HP_loss
    E_HE_aged = E_HE - E_HE_loss
    E_HP_aged = E_HP - E_HP_loss

    N_HE = E_HE / (M_HE * cell_HE.energy)
    N_HP = E_HP / (M_HP * cell_HP.energy)

    SOC_HE_aged = np.full(len(t_soc), 0.9)
    V_HE_aged = np.empty(len(t))
    I_HE_aged = np.empty(len(t))
    P_HE_joule_aged = np.empty(len(t))
    I_HE_joule_aged = np.empty(len(t))

    SOC_HP_aged = np.full(len(t_soc), 0.9)
    V_HP_aged = np.empty(len(t))
    I_HP_aged = np.empty(len(t))
    P_HP_joule_aged = np.empty(len(t))
    I_HP_joule_aged = np.empty(len(t))
    # E_HE_used = np.zeros(len(t_soc))
    # E_HP_used = np.zeros(len(t_soc))


    for i in range(len(t)):
        SOC_HE[i+1] = SOC_HE[i] - (P_HE[i] * (t_soc[i+1]-t_soc[i]) / (E_HE*3.6e6)) if E_HE != 0 else SOC_HE[i]
        SOC_HP[i+1] = SOC_HP[i] - (P_HP[i] * (t_soc[i+1]-t_soc[i]) / (E_HP*3.6e6)) if E_HP != 0 else SOC_HP[i]
        SOC_HE_aged[i+1] = SOC_HE_aged[i] - (P_HE[i] * (t_soc[i+1]-t_soc[i]) / (E_HE_aged*3.6e6)) if E_HE_aged != 0 else SOC_HE_aged[i]
        SOC_HP_aged[i+1] = SOC_HP_aged[i] - (P_HP[i] * (t_soc[i+1]-t_soc[i]) / (E_HP_aged*3.6e6)) if E_HP_aged != 0 else SOC_HP_aged[i]

        # E_HE_used[i+1] = E_HE_used[i] + (P_HE[i]*(t_soc[i+1]-t_soc[i])/3.6e6)
        # E_HP_used[i+1] = E_HP_used[i] + (P_HP[i]*(t_soc[i+1]-t_soc[i])/3.6e6)

        V_HE[i] = M_HE * np.interp(SOC_HE[i], cell_HE.OCV_SOC, cell_HE.OCV)
        V_HP[i] = M_HP * np.interp(SOC_HP[i], cell_HP.OCV_SOC, cell_HP.OCV)
        V_HE_aged[i] = M_HE * np.interp(SOC_HE_aged[i], cell_HE.OCV_SOC, cell_HE.OCV)
        V_HP_aged[i] = M_HP * np.interp(SOC_HP_aged[i], cell_HP.OCV_SOC, cell_HP.OCV)

        I_HE[i] = P_HE[i]/V_HE[i]
        I_HP[i] = P_HP[i]/V_HP[i]
        I_HE_aged[i] = P_HE[i]/V_HE_aged[i]
        I_HP_aged[i] = P_HP[i]/V_HP_aged[i]

        P_HE_joule[i] = cell_HE.resistance * I_HE[i]**2
        P_HP_joule[i] = cell_HP.resistance * I_HP[i]**2
        P_HE_joule_aged[i] = cell_HE.resistance * I_HE_aged[i]**2
        P_HP_joule_aged[i] = cell_HP.resistance * I_HP_aged[i]**2



    # Calculate efficiency
    E_HE_losses = np.max(cumtrapz(P_HE_joule/1000, t/3600, initial=0)) # kWh
    E_HP_losses = np.max(cumtrapz(P_HP_joule/1000, t/3600, initial=0)) # kWh
    E_HE_use = np.max(cumtrapz(P_HE/1000, t/3600, initial=0))
    E_HP_use = np.max(cumtrapz(P_HP/1000, t/3600, initial=0))

    efficiency_HE = (E_HE_use / (E_HE_use + E_HE_losses)) * 100 if E_HE_use !=0 else 0
    efficiency_HP = (E_HP_use / (E_HP_use + E_HP_losses)) * 100 if E_HP_use !=0 else 0
    efficiency = (E_HE_use/(E_HE_use + E_HP_use)) * efficiency_HE + (E_HP_use/(E_HP_use + E_HE_use)) * efficiency_HP

    duration = (time.time() - start_time)*1000
    print(f"Rule-based solution found! \t[{duration:0.2f} ms]")

    # Place results in a dictionary
    result = {
        "t": t,
        "t_soc": t_soc,
        "P": P,
        "P_HE": P_HE,
        "P_HP": P_HP,
        "P_HE_joule": P_HE_joule,
        "P_HP_joule": P_HP_joule,
        "SOC_HE": SOC_HE,
        "SOC_HP": SOC_HP,
        "V_HE": V_HE,
        "V_HP": V_HP,
        "I_HE": I_HE,
        "I_HP": I_HP,
        "M_HE": (V_bus/cell_HE.voltage),
        "M_HP": (V_bus/cell_HP.voltage),
        "N_HE": N_HE,
        "N_HP": N_HP,
        "cost": M_HE*N_HE*cell_HE.cost + M_HP*N_HP*cell_HP.cost,
        "limit": limit[index_min],
        "losses": E_HE_losses + E_HP_losses,
        "efficiency": efficiency,
        "P_HE_joule_aged": P_HE_joule_aged,
        "P_HP_joule_aged": P_HP_joule_aged,
        "SOC_HE_aged": SOC_HE_aged,
        "SOC_HP_aged": SOC_HP_aged,
        "V_HE_aged": V_HE_aged,
        "V_HP_aged": V_HP_aged,
        "I_HE_aged": I_HE_aged,
        "I_HP_aged": I_HP_aged,
        "E_HE": E_HE,
        "E_HP": E_HP,
        "E_HE_aged": E_HE_aged,
        "E_HP_aged": E_HP_aged,
        "I_HE_rated": N_HE*cell_HE.dis_current,
        "I_HP_rated": N_HP*cell_HP.dis_current,
        "E_HE_used": E_HE_use,
        "E_HP_used": E_HP_use,
        "time": duration,
        "method": treshold,
    }

    return result

def treshold_opti(loads, cell_HE, cell_HP, V_bus, cycles, dict_initial=None):
    start_time = time.time()
    if dict_initial == None:
        dict_initial = treshold(loads, cell_HE, cell_HP, V_bus, cycles)

    opti = ca.Opti()

    P = loads[0]["P"].values # W
    t = loads[0]["t"].values # s
    t_soc = np.append(t, t[-1] + (t[-1] - t[-2])) # s

    # Parameters
    M_HE = opti.parameter()
    M_HP = opti.parameter()

    # Decision variables
    P_limit = opti.variable(1, 1)
    P_HE = opti.variable(len(t),1)      # Power from HE battery       [W]
    P_HP = opti.variable(len(t),1)      # Power from HP battery       [W]
    P_HE_joule = opti.variable(len(t),1)      # Joule losses from HE battery       [W]
    P_HP_joule = opti.variable(len(t),1)      # Joule losses from HP battery       [W]
    N_HE = opti.variable(1,1)           # Number of parallel HE strings []
    N_HP = opti.variable(1,1)           # Number of parallel HP strings []
    SOC_HE = opti.variable(len(t)+1, 1)   # State of Charge of HE batttery [0-1]
    SOC_HP = opti.variable(len(t)+1, 1)   # State of Charge of HP batttery [0-1]
    V_HE = opti.variable(len(t), 1)
    V_HP = opti.variable(len(t), 1)
    I_HE = opti.variable(len(t), 1)
    I_HP = opti.variable(len(t), 1)

    P_HE_joule_aged = opti.variable(len(t),1)      # Joule losses from HE battery       [W]
    P_HP_joule_aged = opti.variable(len(t),1)      # Joule losses from HP battery       [W]
    N_HE_aged = opti.variable(1,1)           # Number of parallel HE strings []
    N_HP_aged = opti.variable(1,1)           # Number of parallel HP strings []
    SOC_HE_aged = opti.variable(len(t)+1, 1)   # State of Charge of HE batttery [0-1]
    SOC_HP_aged = opti.variable(len(t)+1, 1)   # State of Charge of HP batttery [0-1]
    V_HE_aged = opti.variable(len(t), 1)
    V_HP_aged = opti.variable(len(t), 1)
    I_HE_aged = opti.variable(len(t), 1)
    I_HP_aged = opti.variable(len(t), 1)

    E_HE_aged = opti.variable(1, 1)
    E_HP_aged = opti.variable(1, 1)
    DOD_HE = opti.variable(1, 1)
    DOD_HP = opti.variable(1, 1)
    E_HE_used = opti.variable(1, 1)
    E_HP_used = opti.variable(1, 1)
    
    # Objective function
    obj = (M_HE*N_HE*cell_HE.cost) + (M_HP*N_HP*cell_HP.cost)

    # Constraintss
    opti.subject_to([
        N_HE > 0, # May not be 0 to avoid division by 0
        N_HP > 0,
        
        SOC_HE <= 0.9,
        SOC_HP <= 0.9,
        SOC_HE >= 0.1,
        SOC_HP >= 0.1,
        SOC_HE_aged <= 0.9,
        SOC_HP_aged <= 0.9,
        SOC_HE_aged >= 0.1,
        SOC_HP_aged >= 0.1,

        E_HE_used > 0,
        E_HP_used > 0,

        E_HE_aged > 0,
        E_HP_aged > 0,

        DOD_HE < 100,
        DOD_HP < 100,
        DOD_HE > 0,
        DOD_HP > 0,

        P_HE_joule == cell_HE.resistance * (I_HE**2), # Calculate HE joule losses (R*I²)
        P_HP_joule == cell_HP.resistance * (I_HP**2), # Calculate HP joule losses (R*I²)
        P_HE_joule_aged == cell_HE.resistance * (I_HE_aged**2), # Calculate HE joule losses (R*I²)
        P_HP_joule_aged == cell_HP.resistance * (I_HP_aged**2), # Calculate HP joule losses (R*I²)

        E_HE_aged == (M_HE*N_HE*cell_HE.energy) * (1 - 0.2 * (cycles[0]/(cell_HE.aging[0]*ca.exp(cell_HE.aging[1]*DOD_HE)+cell_HE.aging[2]*ca.exp(cell_HE.aging[3]*DOD_HE)))),
        E_HP_aged == (M_HP*N_HP*cell_HP.energy) * (1 - 0.2 * (cycles[0]/(cell_HP.aging[0]*ca.exp(cell_HP.aging[1]*DOD_HP)+cell_HP.aging[2]*ca.exp(cell_HP.aging[3]*DOD_HP)))),


        DOD_HE == (E_HE_used/(M_HE*N_HE*cell_HE.energy))*100,
        DOD_HP == (E_HP_used/(M_HP*N_HP*cell_HP.energy))*100,
        ])

    SOC_HE[0] = ca.DM(0.9)
    SOC_HP[0] = ca.DM(0.9)
    SOC_HE_aged[0] = ca.DM(0.9)
    SOC_HP_aged[0] = ca.DM(0.9)
    SOC_TO_OCV_HE = ca.interpolant('LUT', 'bspline', [cell_HE.OCV_SOC], cell_HE.OCV)
    SOC_TO_OCV_HP = ca.interpolant('LUT', 'bspline', [cell_HP.OCV_SOC], cell_HP.OCV)

    for i in range(len(t)):
        opti.subject_to(P_HE[i] == ca.if_else(P[i] > P_limit, P_limit, P[i]))
        opti.subject_to(P_HP[i] == ca.if_else(P[i] > P_limit, P[i]-P_limit, 0))

        opti.subject_to(SOC_HE[i+1] == SOC_HE[i] - ((P_HE[i]*(t_soc[i+1]-t_soc[i]))/((M_HE*N_HE*cell_HE.energy*3.6e6))))
        opti.subject_to(SOC_HP[i+1] == SOC_HP[i] - ((P_HP[i]*(t_soc[i+1]-t_soc[i]))/((M_HP*N_HP*cell_HP.energy*3.6e6))))
        opti.subject_to(SOC_HE_aged[i+1] == SOC_HE_aged[i] - ((P_HE[i]*(t_soc[i+1]-t_soc[i]))/((E_HE_aged*3.6e6))))
        opti.subject_to(SOC_HP_aged[i+1] == SOC_HP_aged[i] - ((P_HP[i]*(t_soc[i+1]-t_soc[i]))/((E_HP_aged*3.6e6))))


        opti.subject_to(V_HE[i] == M_HE*SOC_TO_OCV_HE(SOC_HE[i]))
        opti.subject_to(V_HP[i] == M_HP*SOC_TO_OCV_HP(SOC_HP[i]))
        opti.subject_to(V_HE_aged[i] == M_HE*SOC_TO_OCV_HE(SOC_HE_aged[i]))
        opti.subject_to(V_HP_aged[i] == M_HP*SOC_TO_OCV_HP(SOC_HP_aged[i]))

        opti.subject_to(I_HE[i] == P_HE[i]/V_HE[i])
        opti.subject_to(I_HP[i] == P_HP[i]/V_HP[i])
        opti.subject_to(I_HE_aged[i] == P_HE[i]/V_HE_aged[i])
        opti.subject_to(I_HP_aged[i] == P_HP[i]/V_HP_aged[i])

        opti.subject_to(I_HE[i] + P_HE_joule[i]/V_HE[i] <= cell_HE.dis_current*N_HE)
        opti.subject_to(I_HP[i] + P_HP_joule[i]/V_HP[i] <= cell_HP.dis_current*N_HP)
        opti.subject_to(I_HE_aged[i] + P_HE_joule_aged[i]/V_HE_aged[i] <= cell_HE.dis_current*N_HE)
        opti.subject_to(I_HP_aged[i] + P_HP_joule_aged[i]/V_HP_aged[i] <= cell_HP.dis_current*N_HP)

    opti.subject_to(E_HE_used == sum(P_HE[i] * (t_soc[i+1] - t_soc[i]) / 3.6e6 for i in range(len(t)-1)))
    opti.subject_to(E_HP_used == sum(P_HP[i] * (t_soc[i+1] - t_soc[i]) / 3.6e6 for i in range(len(t)-1)))

    # Set initial values
    opti.set_value(M_HE, V_bus/cell_HE.voltage)
    opti.set_value(M_HP, V_bus/cell_HP.voltage)

    if dict_initial != None:
        opti.set_initial(P_limit, dict_initial["limit"])
        opti.set_initial(P_HE, dict_initial["P_HE"])
        opti.set_initial(P_HP, dict_initial["P_HP"])
        opti.set_initial(N_HE, dict_initial["N_HE"] if dict_initial["N_HE"] != 0 else 1e-9)
        opti.set_initial(N_HP, dict_initial["N_HP"] if dict_initial["N_HP"] != 0 else 1e-9)
        opti.set_initial(SOC_HE, dict_initial["SOC_HE"])
        opti.set_initial(SOC_HP, dict_initial["SOC_HP"])
        opti.set_initial(V_HE, dict_initial["V_HE"])
        opti.set_initial(V_HP, dict_initial["V_HP"])
        opti.set_initial(I_HE, dict_initial["I_HE"])
        opti.set_initial(I_HP, dict_initial["I_HE"])
        opti.set_initial(P_HE_joule, dict_initial["P_HE_joule"])
        opti.set_initial(P_HP_joule, dict_initial["P_HP_joule"])
        opti.set_initial(SOC_HE_aged, dict_initial["SOC_HE_aged"])
        opti.set_initial(SOC_HP_aged, dict_initial["SOC_HP_aged"])
        opti.set_initial(V_HE_aged, dict_initial["V_HE_aged"])
        opti.set_initial(V_HP_aged, dict_initial["V_HP_aged"])
        opti.set_initial(I_HE_aged, dict_initial["I_HE_aged"])
        opti.set_initial(I_HP_aged, dict_initial["I_HE_aged"])
        opti.set_initial(P_HE_joule_aged, dict_initial["P_HE_joule_aged"])
        opti.set_initial(P_HP_joule_aged, dict_initial["P_HP_joule_aged"])
        opti.set_initial(E_HE_aged, dict_initial["E_HE_aged"])
        opti.set_initial(E_HP_aged, dict_initial["E_HP_aged"])
        opti.set_initial(E_HE_used, dict_initial["E_HE_used"])
        opti.set_initial(E_HP_used, dict_initial["E_HP_used"])


    # Start optimization
    opti.minimize(obj)
    options = {"ipopt": {"print_level": 1, "max_iter":3000}} #level5
    opti.solver('ipopt', options)
    try:
        sol = opti.solve()
    except:
        opti.debug.show_infeasibilities()
        print("[ERROR] Optimization Failed!")
        # opti.debug.x_describe(index)
        # opti.debug.g_describe(index)
        exit()


    zero_HE = sol.value(N_HE) < 1e-3    # Check if there are HE cells in the solutions, to remove artifacts
    zero_HP = sol.value(N_HP) < 1e-3    # Check if there are HP cells in the solutions, to remove artifacts

    duration = (time.time() - start_time)*1000
    result = {
        "t": t,
        "t_soc": t_soc,
        "P": P,
        "P_HE": sol.value(P_HE),
        "P_HP": sol.value(P_HP),
        "P_HE_joule": sol.value(P_HE_joule),
        "P_HP_joule": sol.value(P_HP_joule),
        "SOC_HE": np.full(len(t_soc), 0.9) if zero_HE else sol.value(SOC_HE),
        "SOC_HP": np.full(len(t_soc), 0.9) if zero_HP else sol.value(SOC_HP),
        "V_HE": np.full(len(t), sol.value(V_HE[0])) if zero_HE else sol.value(V_HE),
        "V_HP": np.full(len(t), sol.value(V_HP[0])) if zero_HP else sol.value(V_HP),
        "I_HE": sol.value(I_HE),
        "I_HP": sol.value(I_HP),
        "M_HE": sol.value(M_HE),
        "M_HP": sol.value(M_HP),
        "N_HE": sol.value(N_HE),
        "N_HP": sol.value(N_HP),
        "cost": sol.value(M_HE) * sol.value(N_HE) * cell_HE.cost + sol.value(M_HP) * sol.value(N_HP) * cell_HP.cost,
        "losses": 0 + 0,
        "efficiency": 0,
        "P_HE_joule_aged": sol.value(P_HE_joule_aged),
        "P_HP_joule_aged": sol.value(P_HP_joule_aged),
        "SOC_HE_aged": np.full(len(t_soc), 0.9) if zero_HE else sol.value(SOC_HE_aged),
        "SOC_HP_aged": np.full(len(t_soc), 0.9) if zero_HP else sol.value(SOC_HP_aged),
        "V_HE_aged": np.full(len(t), sol.value(V_HE[0])) if zero_HE else sol.value(V_HE_aged),
        "V_HP_aged": np.full(len(t), sol.value(V_HP[0])) if zero_HP else sol.value(V_HP_aged),
        "I_HE_aged": sol.value(I_HE_aged),
        "I_HP_aged": sol.value(I_HP_aged),
        "E_HE": sol.value(M_HE) * sol.value(N_HE) * cell_HE.energy,
        "E_HP": sol.value(M_HP) * sol.value(N_HP) * cell_HP.energy,
        "E_HE_aged": sol.value(E_HE_aged),
        "E_HP_aged": sol.value(E_HP_aged),
        "I_HE_rated": sol.value(N_HE)*cell_HE.dis_current,
        "I_HP_rated": sol.value(N_HP)*cell_HP.dis_current,
        "DOD_HE": sol.value(DOD_HE),
        "DOD_HP": sol.value(DOD_HP),
        "time": duration,
        "method": "optimal"
        #"limit": None
    }

    return result

