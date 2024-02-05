import casadi as ca
import numpy as np
from scipy.integrate import cumtrapz
import time

def optimal(loads, cell_HE, cell_HP, V_bus, cycles=[0], bool_intercharge=False, dict_initial=None):
    start_time = time.time()
    opti = ca.Opti()

    P = loads[0]["P"].values # W
    t = loads[0]["t"].values # s
    t_soc = np.append(t, t[-1] + (t[-1] - t[-2])) # s

    # Parameters
    M_HE = opti.parameter()
    M_HP = opti.parameter()

    # Decision variables
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

    # Objective function
    obj = ((M_HE*N_HE*cell_HE.cost) + (M_HP*N_HP*cell_HP.cost)) # + # Total cost of battery cells
           #100000 * ((SOC_HE[-1]-0.1) + (SOC_HP[-1]-0.1)) + # End at minimum SOC
          # 1000 * (ca.sum1(P_HE_joule)/np.sum(P)) + (ca.sum1(P_HP_joule)/np.sum(P))) # Minimize joule losses

    # Constraints
    opti.subject_to([
        P_HE + P_HP == P,
        N_HE >= 0, # May not be 0 to avoid division by 0
        N_HP >= 0,
        
        SOC_HE <= 0.9,
        SOC_HP <= 0.9,
        SOC_HE >= 0.1,
        SOC_HP >= 0.1,

        # SOC_HE[-1] == ca.if_else(N_HE == 0, 0.9, 0.1), # <- GIVES ERROR
        # SOC_HP[-1] == ca.if_else(N_HP == 0, 0.9, 0.1),

        P_HE_joule == cell_HE.resistance * (I_HE**2), # Calculate HE joule losses (R*I²)
        P_HP_joule == cell_HP.resistance * (I_HP**2), # Calculate HP joule losses (R*I²)
        ])
    
    if not bool_intercharge:
        opti.subject_to([
            P_HE >= 0,
            P_HP >= 0,
        ])

    SOC_HE[0] = ca.DM(0.9)
    SOC_HP[0] = ca.DM(0.9)
    SOC_TO_OCV_HE = ca.interpolant('LUT', 'bspline', [cell_HE.OCV_SOC], cell_HE.OCV)
    SOC_TO_OCV_HP = ca.interpolant('LUT', 'bspline', [cell_HP.OCV_SOC], cell_HP.OCV)

    for i in range(len(t)):
        opti.subject_to(SOC_HE[i+1] == ca.if_else(N_HE == 0, SOC_HE[i], SOC_HE[i] - ((P_HE[i]*(t_soc[i+1]-t_soc[i]))/((M_HE*N_HE*cell_HE.energy*3.6e6)))))
        opti.subject_to(SOC_HP[i+1] == ca.if_else(N_HP == 0, SOC_HP[i], SOC_HP[i] - ((P_HP[i]*(t_soc[i+1]-t_soc[i]))/((M_HP*N_HP*cell_HP.energy*3.6e6)))))

        opti.subject_to(V_HE[i] == M_HE*SOC_TO_OCV_HE(SOC_HE[i]))
        opti.subject_to(V_HP[i] == M_HP*SOC_TO_OCV_HP(SOC_HP[i]))

        opti.subject_to(I_HE[i] == P_HE[i]/V_HE[i])
        opti.subject_to(I_HP[i] == P_HP[i]/V_HP[i])

        opti.subject_to(I_HE[i] + P_HE_joule[i]/V_HE[i] <= cell_HE.dis_current*N_HE)
        opti.subject_to(I_HP[i] + P_HP_joule[i]/V_HP[i] <= cell_HP.dis_current*N_HP)

    # Set initial values
    opti.set_value(M_HE, V_bus/cell_HE.voltage)
    opti.set_value(M_HP, V_bus/cell_HP.voltage)

    if dict_initial != None:
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

    print(sol.value(obj))
    print(f"M_HE: {sol.value(M_HE)} & N_HE: {sol.value(N_HE)}")
    print(f"M_HP: {sol.value(M_HP)} & N_HE: {sol.value(N_HP)}")

    E_HE_losses = np.max(cumtrapz(sol.value(P_HE_joule)/1000, t/3600, initial=0)) # kWh
    E_HP_losses = np.max(cumtrapz(sol.value(P_HP_joule)/1000, t/3600, initial=0)) # kWh
    E_HE_used = np.max(cumtrapz(sol.value(P_HE)/1000, t/3600, initial=0))
    E_HP_used = np.max(cumtrapz(sol.value(P_HP)/1000, t/3600, initial=0))

    efficiency_HE = (E_HE_used / (E_HE_used + E_HE_losses)) * 100
    efficiency_HP = (E_HP_used / (E_HP_used + E_HP_losses)) * 100
    efficiency = (E_HE_used/(E_HE_used + E_HP_used)) * efficiency_HE + (E_HP_used/(E_HP_used + E_HE_used)) * efficiency_HP


    # Place results in a dictionary
    zero_HE = sol.value(N_HE) < 1e-3    # Check if there are HE cells in the solutions, to remove artifacts
    zero_HP = sol.value(N_HP) < 1e-3    # Check if there are HP cells in the solutions, to remove artifacts

    duration = (time.time() - start_time)*1000
    print(f"Optimal solution found! \t[{duration:0.2f} ms]")

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
        "losses": E_HE_losses + E_HP_losses,
        "efficiency": efficiency,
        "P_HE_joule_aged": sol.value(P_HE_joule),
        "P_HP_joule_aged": sol.value(P_HP_joule),
        "SOC_HE_aged": np.full(len(t_soc), 0.9) if zero_HE else sol.value(SOC_HE),
        "SOC_HP_aged": np.full(len(t_soc), 0.9) if zero_HP else sol.value(SOC_HP),
        "V_HE_aged": np.full(len(t), sol.value(V_HE[0])) if zero_HE else sol.value(V_HE),
        "V_HP_aged": np.full(len(t), sol.value(V_HP[0])) if zero_HP else sol.value(V_HP),
        "I_HE_aged": sol.value(I_HE),
        "I_HP_aged": sol.value(I_HP),
        "E_HE": sol.value(M_HE) * sol.value(N_HE) * cell_HE.energy,
        "E_HP": sol.value(M_HP) * sol.value(N_HP) * cell_HP.energy,
        "E_HE_aged": sol.value(M_HE) * sol.value(N_HE) * cell_HE.energy,
        "E_HP_aged": sol.value(M_HP) * sol.value(N_HP) * cell_HP.energy,
        "I_HE_rated": sol.value(N_HE)*cell_HE.dis_current,
        "I_HP_rated": sol.value(N_HP)*cell_HP.dis_current,
        "time": duration,
        "method": "optimal"
        #"limit": None
    }

    return result





def optimal_aging(loads, cell_HE, cell_HP, V_bus, cycles=[0], bool_intercharge=False, dict_initial=None):
    start_time = time.time()
    opti = ca.Opti()

    P = loads[0]["P"].values # W
    t = loads[0]["t"].values # s
    t_soc = np.append(t, t[-1] + (t[-1] - t[-2])) # s

    # Parameters
    M_HE = opti.parameter()
    M_HP = opti.parameter()

    # Decision variables
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
    obj = ((M_HE*N_HE*cell_HE.cost) + (M_HP*N_HP*cell_HP.cost) + # Total cost of battery cells
           #100000 * ((SOC_HE[-1]-0.1) + (SOC_HP[-1]-0.1)) + # End at minimum SOC
           1000 * (ca.sum1(P_HE_joule)/np.sum(P)) + (ca.sum1(P_HP_joule)/np.sum(P)))  # Minimize joule losses
           #+ 10 * ca.sum1(P_HE)/len(t)) # Minimize average HE power

    # Constraints
    opti.subject_to([
        P_HE + P_HP == P,
        N_HE >= 0, # May not be 0 to avoid division by 0
        N_HP >= 0,
        
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

        E_HE_aged == (M_HE*N_HE*cell_HE.energy) * (1 - 0.2 * (cycles[0]/(cell_HE.aging[0]*ca.exp(cell_HE.aging[1]*DOD_HE)+cell_HE.aging[2]*ca.exp(cell_HE.aging[3]*DOD_HE)))),
        E_HP_aged == (M_HP*N_HP*cell_HP.energy) * (1 - 0.2 * (cycles[0]/(cell_HP.aging[0]*ca.exp(cell_HP.aging[1]*DOD_HP)+cell_HP.aging[2]*ca.exp(cell_HP.aging[3]*DOD_HP)))),


        DOD_HE == (E_HE_used/(M_HE*N_HE*cell_HE.energy))*100,
        DOD_HP == (E_HP_used/(M_HP*N_HP*cell_HP.energy))*100,

        # SOC_HE[-1] == ca.if_else(N_HE == 0, 0.9, 0.1), # <- GIVES ERROR
        # SOC_HP[-1] == ca.if_else(N_HP == 0, 0.9, 0.1),

        P_HE_joule == cell_HE.resistance * (I_HE**2), # Calculate HE joule losses (R*I²)
        P_HP_joule == cell_HP.resistance * (I_HP**2), # Calculate HP joule losses (R*I²)
        P_HE_joule_aged == cell_HE.resistance * (I_HE_aged**2), # Calculate HE joule losses (R*I²)
        P_HP_joule_aged == cell_HP.resistance * (I_HP_aged**2), # Calculate HP joule losses (R*I²)
        ])
    
    if not bool_intercharge:
        opti.subject_to([
            P_HE >= 0,
            P_HP >= 0,
        ])

    SOC_HE[0] = ca.DM(0.9)
    SOC_HP[0] = ca.DM(0.9)
    SOC_HE_aged[0] = ca.DM(0.9)
    SOC_HP_aged[0] = ca.DM(0.9)
    SOC_TO_OCV_HE = ca.interpolant('LUT', 'bspline', [cell_HE.OCV_SOC], cell_HE.OCV)
    SOC_TO_OCV_HP = ca.interpolant('LUT', 'bspline', [cell_HP.OCV_SOC], cell_HP.OCV)

    for i in range(len(t)):
        opti.subject_to(SOC_HE[i+1] == ca.if_else(N_HE == 0, SOC_HE[i], SOC_HE[i] - ((P_HE[i]*(t_soc[i+1]-t_soc[i]))/((M_HE*N_HE*cell_HE.energy*3.6e6)))))
        opti.subject_to(SOC_HP[i+1] == ca.if_else(N_HP == 0, SOC_HP[i], SOC_HP[i] - ((P_HP[i]*(t_soc[i+1]-t_soc[i]))/((M_HP*N_HP*cell_HP.energy*3.6e6)))))
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
        
        opti.set_initial(V_HE_aged, dict_initial["V_HE_aged"])
        opti.set_initial(V_HP_aged, dict_initial["V_HP_aged"])
        opti.set_initial(I_HE_aged, dict_initial["I_HE_aged"])
        opti.set_initial(I_HP_aged, dict_initial["I_HE_aged"])
        opti.set_initial(P_HE_joule_aged, dict_initial["P_HE_joule_aged"])
        opti.set_initial(P_HP_joule_aged, dict_initial["P_HP_joule_aged"])
        opti.set_initial(E_HE_aged, dict_initial["E_HE_aged"])
        opti.set_initial(E_HP_aged, dict_initial["E_HP_aged"])


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

    print(sol.value(obj))
    print(f"M_HE: {sol.value(M_HE)} & N_HE: {sol.value(N_HE)}")
    print(f"M_HP: {sol.value(M_HP)} & N_HE: {sol.value(N_HP)}")

    E_HE_losses = np.max(cumtrapz(sol.value(P_HE_joule)/1000, t/3600, initial=0)) # kWh
    E_HP_losses = np.max(cumtrapz(sol.value(P_HP_joule)/1000, t/3600, initial=0)) # kWh
    E_HE_used = np.max(cumtrapz(sol.value(P_HE)/1000, t/3600, initial=0))
    E_HP_used = np.max(cumtrapz(sol.value(P_HP)/1000, t/3600, initial=0))

    efficiency_HE = (E_HE_used / (E_HE_used + E_HE_losses)) * 100
    efficiency_HP = (E_HP_used / (E_HP_used + E_HP_losses)) * 100
    efficiency = (E_HE_used/(E_HE_used + E_HP_used)) * efficiency_HE + (E_HP_used/(E_HP_used + E_HE_used)) * efficiency_HP


    # Place results in a dictionary
    zero_HE = sol.value(N_HE) < 1e-3    # Check if there are HE cells in the solutions, to remove artifacts
    zero_HP = sol.value(N_HP) < 1e-3    # Check if there are HP cells in the solutions, to remove artifacts

    duration = (time.time() - start_time)*1000
    print(f"Optimal solution found! \t[{duration:0.2f} ms]")

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
        "losses": E_HE_losses + E_HP_losses,
        "efficiency": efficiency,
        "P_HE_joule_aged": sol.value(P_HE_joule_aged),
        "P_HP_joule_aged": sol.value(P_HP_joule_aged),
        "SOC_HE_aged": sol.value(SOC_HE_aged),
        "SOC_HP_aged": sol.value(SOC_HP_aged),
        "V_HE_aged": sol.value(V_HE_aged),
        "V_HP_aged": sol.value(V_HP_aged),
        "I_HE_aged": sol.value(I_HE_aged),
        "I_HP_aged": sol.value(I_HP_aged),
        "E_HE": sol.value(M_HE) * sol.value(N_HE) * cell_HE.energy,
        "E_HP": sol.value(M_HP) * sol.value(N_HP) * cell_HP.energy,
        "E_HE_aged": sol.value(E_HE_aged),
        "E_HP_aged": sol.value(E_HP_aged),
        "I_HE_rated": sol.value(N_HE)*cell_HE.dis_current,
        "I_HP_rated": sol.value(N_HP)*cell_HP.dis_current,
        "time": duration,
        "method": "optimal"
        #"limit": None
    }

    return result