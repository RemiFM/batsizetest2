import streamlit as st
from typing import NamedTuple 
from typing import List
import numpy as np
import pandas as pd

from funcs import st_plot
from methods import monotype
from methods import treshold
from methods import optimal
from funcs import plotting

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

st.set_page_config(
    page_title="Battery Sizing Tool",
    page_icon=":battery:",
    layout="wide",
)

st.markdown("""
        <style>
               .block-container {
                    padding-top: 1rem;
                    padding-bottom: 0rem;
                    padding-left: 5rem;
                    padding-right: 5rem;
                }
        </style>
        """, unsafe_allow_html=True)

hex_color = ["#66c298", "#fc8d62", "#8da0cb"]

st.markdown(f"<h2 style='text-align: center; color: {hex_color[0]};'>Battery Sizing Tool 🔋</h2>", unsafe_allow_html=True)
tabs = st.tabs([":one: Inputs", ":two: Results", ":three: Comparison"])
placeholders = [tabs[1].empty(), tabs[2].empty()]
for i in placeholders: i.warning("Click the 'Start Calculation' button to generate results", icon="🎯")



hcols = tabs[0].columns([4,5], gap="large")
cols = hcols[0].columns(2, gap="medium")

#cols[0].markdown(f"<div style='background-color: {hex_color[2]}; color: white; padding: 8px; text-align: center; font-weight: bold; border-radius: 10px;'>High Energy Cell</div>", unsafe_allow_html=True)
#cols[1].markdown(f"<div style='background-color: {hex_color[1]}; color: white; padding: 8px; text-align: center; font-weight: bold; border-radius: 10px;'>High Power Cell</div>", unsafe_allow_html=True)
cols[0].markdown(f"<div style='border: 4px solid {hex_color[2]}; padding: 8px; text-align: center; font-weight: bold; color: {hex_color[2]}; border-radius: 10px;'>High Energy Cell</div>", unsafe_allow_html=True)
cols[1].markdown(f"<div style='border: 4px solid {hex_color[1]}; padding: 8px; text-align: center; font-weight: bold; color: {hex_color[1]}; border-radius: 10px;'>High Power Cell</div>", unsafe_allow_html=True)

cols[0].write("")
cols[1].write("")

cols = hcols[0].columns(2, gap="medium")

CELL_TECHS = ["NMC - Nickel Magnesium Cobalt", "LTO - Lithium Titanate"]
select_HE = cols[0].selectbox("Battery Cell Technology", CELL_TECHS, index=0)
select_HP = cols[1].selectbox("Battery Cell Technology", CELL_TECHS, index=1)

capacity_HE = cols[0].number_input("Rated Capacity _(Ah)_", value=50, min_value=0, step=10)
#scols = cols[0].columns(2)

cell_HE = BatteryCell(
    capacity    = capacity_HE,
    voltage     = 3.67,
    dis_rate    = cols[0].number_input("Discharge C-rate _(A/Ah)_", value=1.0, min_value=0.0, step=0.25),
    chg_rate    = cols[1].number_input("Charge C-rate _(A/Ah)_", value=1.0, min_value=0.0, step=0.25),
    resistance  = cols[0].number_input("Internal Resistance _(mΩ)_", value=1.5, min_value=0.0, step=0.1)/1000,
    weight      = cols[1].number_input("Cell Weight _(g)_", value=0.885*1000, min_value=0.0, step=0.1),
    cost_spec   = cols[0].number_input("Specific Cost _(€/kWh)_", value=150, min_value=0, step=25),
    OCV         = [3.427, 3.508, 3.588, 3.621, 3.647, 3.684, 3.761, 3.829, 3.917, 4.019, 4.135],
    OCV_SOC     = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1],
    aging       = [694700, -0.1770, 52790, -0.0356],
)

capacity_HP = cols[1].number_input("Rated Capacity _(Ah)_", value=23, min_value=0, step=10)
#scols = cols[1].columns(2)

cell_HP = BatteryCell(
    capacity    = capacity_HP,
    voltage     = 2.3,
    dis_rate    = cols[0].number_input("Discharge C-rate _(A/Ah)_", value=4.0, min_value=0.0, step=0.25),
    chg_rate    = cols[1].number_input("Charge C-rate _(A/Ah)_", value=4.0, min_value=0.0, step=0.25),
    resistance  = cols[0].number_input("Internal Resistance _(mΩ)_", value=1.1, min_value=0.0, step=0.1)/1000,
    weight      = cols[1].number_input("Cell Weight _(g)_", value=0.55*1000, min_value=0.0, step=0.1),
    cost_spec   = cols[1].number_input("Specific Cost _(€/kWh)_", value=380, min_value=0, step=25),
    OCV         = [2.067, 2.113, 2.151, 2.183, 2.217, 2.265, 2.326, 2.361, 2.427, 2.516, 2.653],
    OCV_SOC     = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1],
    aging       = [6881000, -0.1950, 426500, -0.0418],
)

hcols[0].divider()
cols = hcols[0].columns([2, 1], gap="medium")
V_bus = cols[0].number_input("Nominal Battery Voltage _(V)_", value=1000, min_value=0, step=100)
bool_aging = cols[1].checkbox("Enable Cell Degradation", value=True)
bool_charge = cols[1].checkbox("Allow Intercharging", value=False)

run = tabs[0].button("Start Calculation...", type="primary", use_container_width=True)

load_files = [hcols[1].file_uploader("Upload a load profile", type=["csv"], key=137)]
cycles = [hcols[1].number_input("Load cycles during lifetime", min_value=0, value=3650, key=120, disabled=not bool_aging)]

if load_files[0] is not None:
    loads = [pd.read_csv(load_files[0])]
else:
    loads = [pd.read_csv("loads/sinc_1.csv")]

fig = st_plot.plot_load_profiles(loads, 420)
hcols[1].altair_chart(fig, use_container_width=True)

if run:
    for i in placeholders: i.empty()
    with st.spinner("Calculation in progress..."):
        st.toast("Calculating monotype HE solution...", icon="⌛")
        dict_mono_HE = monotype.monotype2(loads=loads, cell=cell_HE, V_bus=V_bus, cycles=cycles if bool_aging else [0]*len(loads))
        st.toast(f"[{dict_mono_HE['time']/1000:,.2f}s] Monotype HE solution found!", icon="✅")

        st.toast("Calculating monotype HP solution...", icon="⌛")
        dict_mono_HP = monotype.monotype2(loads=loads, cell=cell_HP, V_bus=V_bus, cycles=cycles if bool_aging else [0]*len(loads))
        st.toast(f"[{dict_mono_HP['time']/1000:,.2f}s] Monotype HP solution found!", icon="✅")

        st.toast("Calculating rule-based hybrid solution...", icon="⌛")
        dict_tresh = treshold.treshold_opti(loads=loads, cell_HE=cell_HE, cell_HP=cell_HP, V_bus=V_bus, cycles=cycles)
        st.toast(f"[{dict_tresh['time']/1000:,.2f}s] Rule-based hybrid solution found!", icon="✅")

        st.toast("Calculating optimal hybrid solution...", icon="⌛")
        dict_opti = optimal.optimal_aging(loads, cell_HE, cell_HP, V_bus, cycles=cycles, bool_intercharge=False, dict_initial=dict_tresh)
        st.toast(f"[{dict_opti['time']/1000:,.2f}s] Optimal hybrid solution found!", icon="✅")

        if bool_charge:
            st.toast("Calculating optimal hybrid solution with intercharging...", icon="⌛")
            dict_opti2 = optimal.optimal_aging(loads, cell_HE, cell_HP, V_bus, cycles=cycles, bool_intercharge=True, dict_initial=dict_tresh)
            st.toast(f"[{dict_opti2['time']/1000:,.2f}s] Optimal hybrid solution with intercharging found!", icon="✅")

        time_tot = dict_mono_HE['time'] + dict_mono_HP['time'] + dict_tresh['time'] + dict_opti['time'] + dict_opti2['time'] if bool_charge else dict_mono_HE['time'] + dict_mono_HP['time'] + dict_tresh['time'] + dict_opti['time']
        st.toast(f"All results found in **{time_tot/1000:.2f}s**!", icon="✅")

        cols = tabs[1].columns(3 if bool_charge else 2, gap="medium")

        cols[0].markdown(f"<div style='border: 4px solid {hex_color[0]}; padding: 8px; text-align: center; font-weight: bold; color: {hex_color[0]}; border-radius: 10px;'>Rule-Based Hybrid</div>", unsafe_allow_html=True)
        cols[0].write("")
        cols[0].altair_chart(st_plot.plot_powers(dict_tresh), use_container_width=True)
        cols[0].altair_chart(st_plot.plot_SOC(dict_tresh), use_container_width=True)

        cols[1].markdown(f"<div style='border: 4px solid {hex_color[0]}; padding: 8px; text-align: center; font-weight: bold; color: {hex_color[0]}; border-radius: 10px;'>Optimal Hybrid</div>", unsafe_allow_html=True)
        cols[1].write("")
        cols[1].altair_chart(st_plot.plot_powers(dict_opti), use_container_width=True)
        cols[1].altair_chart(st_plot.plot_SOC(dict_opti), use_container_width=True)
        

        if bool_charge:
            cols[2].markdown(f"<div style='border: 4px solid {hex_color[0]}; padding: 8px; text-align: center; font-weight: bold; color: {hex_color[0]}; border-radius: 10px;'>Optimal Hybrid w/ Intercharging</div>", unsafe_allow_html=True)
            cols[2].write("")
            cols[2].altair_chart(st_plot.plot_powers(dict_opti2), use_container_width=True)
            cols[2].altair_chart(st_plot.plot_SOC(dict_opti2), use_container_width=True)

        

        cols = tabs[2].columns(4 if bool_charge else 3, gap="medium")

        cols[0].markdown(f"<div style='border: 4px solid {hex_color[0]}; padding: 8px; text-align: center; font-weight: bold; color: {hex_color[0]}; border-radius: 10px;'>Monotype</div>", unsafe_allow_html=True)
        cols[0].write("")
        cols[0].metric("Total Cost", value=f"€ {min(dict_mono_HE['cost'], dict_mono_HP['cost']):,.2f}")

        cols[1].markdown(f"<div style='border: 4px solid {hex_color[0]}; padding: 8px; text-align: center; font-weight: bold; color: {hex_color[0]}; border-radius: 10px;'>Rule-Based Hybrid</div>", unsafe_allow_html=True)
        cols[1].write("")
        cols[1].metric("Total Cost", value=f"€ {dict_tresh['cost']:,.2f}")

        cols[2].markdown(f"<div style='border: 4px solid {hex_color[0]}; padding: 8px; text-align: center; font-weight: bold; color: {hex_color[0]}; border-radius: 10px;'>Optimal Hybrid</div>", unsafe_allow_html=True)
        cols[2].write("")
        cols[2].metric("Total Cost", value=f"€ {dict_opti['cost']:,.2f}")

        if bool_charge:
            cols[3].markdown(f"<div style='border: 4px solid {hex_color[0]}; padding: 8px; text-align: center; font-weight: bold; color: {hex_color[0]}; border-radius: 10px;'>Optimal Hybrid w/ Intercharging</div>", unsafe_allow_html=True)
            cols[3].write("")
            cols[3].metric("Total Cost", value=f"€ {dict_opti2['cost']:,.2f}")

