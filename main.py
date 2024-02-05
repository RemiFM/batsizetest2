import numpy as np
import pandas as pd
from typing import NamedTuple 
from typing import List
import casadi as ca
import csv

from methods import monotype
from methods import treshold
from methods import optimal
from funcs import plotting

## -- Class definition
class BatteryCell(NamedTuple):  # Class for defining battery cells
    capacity:   float   # Rated capacity (Ah)
    voltage:    float   # Nominal voltage (V)
    dis_rate:   float   # Maximum discharge rate (A/Ah)
    chg_rate:   float   # Maximum charge rate (A/Ah)
    resistance: float   # Internal resistance (Ohm)
    weight:     float   # Cell weight (kg)
    cost_spec:  float   # Specific cost (€/kWh)
    OCV: List[float]        # Open-circuit voltage (V)
    OCV_SOC: List[float]    # State of Charge (0-1)
    aging: List[float]      # fitting parameters for aging model (a, b, c, d)

    @property
    def energy(self):       # Energy Capacity (kWh)
        return self.capacity * self.voltage / 1000
    
    @property
    def cost(self):         # Cost per cell (€)
        return self.energy * self.cost_spec
    
    @property
    def dis_current(self):  # Maximum discharge current (A)
        return self.dis_rate * self.capacity
    
    @property
    def chg_current(self):  # Maximum charge current (A)
        return self.chg_rate * self.capacity

cell_HE = BatteryCell(
    capacity    = 50,
    voltage     = 3.67,
    dis_rate    = 1,
    chg_rate    = 1,
    cost_spec   = 150,
    resistance  = 1.5 / 1000,
    weight      = 1000*0.885, 
    OCV         = [3.427, 3.508, 3.588, 3.621, 3.647, 3.684, 3.761, 3.829, 3.917, 4.019, 4.135],
    OCV_SOC     = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1],
    aging       = [694700, -0.1770, 52790, -0.0356],
)

cell_HP = BatteryCell(
    capacity    = 23,
    voltage     = 2.3,
    dis_rate    = 4,
    chg_rate    = 4,
    cost_spec   = 380,
    resistance  = 1.1 / 1000, 
    weight      = 1000*0.55,
    OCV         = [2.067, 2.113, 2.151, 2.183, 2.217, 2.265, 2.326, 2.361, 2.427, 2.516, 2.653],
    OCV_SOC     = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1],
    aging       = [6881000, -0.1950, 426500, -0.0418],
) 

V_bus = 2500 # Nominal pack voltage (V)
loads = [pd.read_csv("loads/tug_boat_1.csv"), pd.read_csv("loads/tug_boat_2.csv")]
cycles = [10000, 20000] #10950 & 520

# ----------------
# Calculate monotype
plotting.plot_inputs(loads)
dict_mono_test = monotype.monotype_multi(loads, cell_HE, V_bus, cycles)
print(f"{dict_mono_test['time']:,.2f} ms")
plotting.plot_multiple(dict_mono_test)
quit()



# Calculate rule-based treshold sizing
dict_mono_HE = monotype.monotype2(loads, cell_HE, V_bus, cycles=cycles)
dict_treshold = treshold.treshold(loads, cell_HE, cell_HP, V_bus, cycles=cycles)
dict_treshold_opti = treshold.treshold_opti(loads, cell_HE, cell_HP, V_bus, cycles=cycles, dict_initial=dict_treshold)
print("treshold finished")
dict_opti = optimal.optimal_aging(loads, cell_HE, cell_HP, V_bus, cycles=cycles, bool_intercharge=False, dict_initial=dict_treshold_opti)
dict_opti2 = optimal.optimal_aging(loads, cell_HE, cell_HP, V_bus, cycles=cycles, bool_intercharge=True, dict_initial=dict_opti)

print(f"Monotype HE:\t\t\t€ {dict_mono_HE['cost']:,.2f}\t\t{dict_mono_HE['M']:.2f} x {dict_mono_HE['N']:.2f}\t\t-\t\t\t{dict_mono_HE['time']:.2f} ms")
print(f"Rule-based:\t\t\t€ {dict_treshold['cost']:,.2f}\t\t{dict_treshold['M_HE']:.2f} x {dict_treshold['N_HE']:.2f}\t\t{dict_treshold['M_HP']:.2f} x {dict_treshold['N_HP']:.2f}\t\t{dict_treshold['time']:.2f} ms")
print(f"Rule-based:\t\t\t€ {dict_treshold_opti['cost']:,.2f}\t\t{dict_treshold_opti['M_HE']:.2f} x {dict_treshold_opti['N_HE']:.2f}\t\t{dict_treshold_opti['M_HP']:.2f} x {dict_treshold_opti['N_HP']:.2f}\t\t{dict_treshold_opti['time']:.2f} ms")
print(f"Optimised (no intercharging):\t€ {dict_opti['cost']:,.2f}\t\t{dict_opti['M_HE']:.2f} x {dict_opti['N_HE']:.2f}\t\t{dict_opti['M_HP']:.2f} x {dict_opti['N_HP']:.2f}\t\t{dict_opti['time']:.2f} ms")
print(f"Optimised (intercharging):\t€ {dict_opti2['cost']:,.2f}\t\t{dict_opti2['M_HE']:.2f} x {dict_opti2['N_HE']:.2f}\t\t{dict_opti2['M_HP']:.2f} x {dict_opti2['N_HP']:.2f}\t\t{dict_opti2['time']:.2f} ms")

print(f"\nDOD_HE: {dict_treshold_opti['DOD_HE']:.2f}%\t\tDOD_HP: {dict_treshold_opti['DOD_HP']:.2f}%")

#plotting.plot_all(dict_treshold)
#plotting.plot_soc(dict_opti)
#plotting.plot_soc(dict_opti)
plotting.plot_all(dict_opti2)
quit()

#print("\n\nMethod\t\t\t\tTotal cost\t\tHE configuration\tHP configuration\tEfficiency\tLosses")
#print(f"Rule-based:\t\t\t€ {dict_treshold['cost']:,.2f}\t\t{dict_treshold['M_HE']:.2f} x {dict_treshold['N_HE']:.2f}\t\t{dict_treshold['M_HP']:.2f} x {dict_treshold['N_HP']:.2f}\t\t{dict_treshold['efficiency']:.2f}%\t\t{dict_treshold['losses']:.2f} kWh")
#plotting.plot_all(dict_treshold) test



# Calculate optimized sizing (no intercharging)
dict_opti = optimal.optimal(loads, cell_HE, cell_HP, V_bus, cycles=cycles, bool_intercharge=False, dict_initial=dict_treshold)
dict_opti2 = optimal.optimal(loads, cell_HE, cell_HP, V_bus, cycles=cycles, bool_intercharge=True, dict_initial=dict_opti)

print("\n\n--------------------------------------------------------------------------------------------------------------------")
print("Method\t\t\t\tTotal cost\t\tHE configuration\tHP configuration\tCalculation")
print("--------------------------------------------------------------------------------------------------------------------")
print(f"Monotype HE:\t\t\t€ {dict_mono_HE['cost']:,.2f}\t\t{dict_mono_HE['M']:.2f} x {dict_mono_HE['N']:.2f}\t\t-\t\t\t{dict_mono_HE['time']:.2f} ms")
print(f"Monotype HP:\t\t\t€ {dict_mono_HP['cost']:,.2f}\t\t-\t\t\t{dict_mono_HP['M']:.2f} x {dict_mono_HP['N']:.2f}\t\t{dict_mono_HP['time']:.2f} ms")
print(f"Rule-based:\t\t\t€ {dict_treshold['cost']:,.2f}\t\t{dict_treshold['M_HE']:.2f} x {dict_treshold['N_HE']:.2f}\t\t{dict_treshold['M_HP']:.2f} x {dict_treshold['N_HP']:.2f}\t\t{dict_treshold['time']:.2f} ms")
print(f"Optimised (no intercharging):\t€ {dict_opti['cost']:,.2f}\t\t{dict_opti['M_HE']:.2f} x {dict_opti['N_HE']:.2f}\t\t{dict_opti['M_HP']:.2f} x {dict_opti['N_HP']:.2f}\t\t{dict_opti['time']:.2f} ms")
print(f"Optimised (intercharging):\t€ {dict_opti2['cost']:,.2f}\t\t{dict_opti2['M_HE']:.2f} x {dict_opti2['N_HE']:.2f}\t\t{dict_opti2['M_HP']:.2f} x {dict_opti2['N_HP']:.2f}\t\t{dict_opti2['time']:.2f} ms")
print("--------------------------------------------------------------------------------------------------------------------")




plotting.plot_all(dict_opti)
#plotting.compare_power(dict_mono_HE, dict_mono_HP, dict_treshold, dict_opti, dict_opti2)
