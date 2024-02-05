import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

def plot_all(dict_results):
    colors = sns.color_palette("Set2", 5)
    # -- Create Plots --
    # Plot P
    plt.subplot(3, 2, 1)
    plt.plot(dict_results["t"]/3600, dict_results["P"]/1e6, label="Power Demand", color=colors[0] if "P_HE" in dict_results else colors[1], linewidth=1.5)
    plt.fill_between(dict_results["t"]/3600, dict_results["P"]/1e6, 0, label=None, color=colors[0], alpha=0.35)

    if "P_HE" in dict_results:
        plt.plot(dict_results["t"]/3600, dict_results["P_HE"]/1e6, label="High Energy Pack", color=colors[2], linewidth=1.5)
        plt.plot(dict_results["t"]/3600, dict_results["P_HP"]/1e6, label="High Power Pack", color=colors[1], linewidth=1.5)

        plt.fill_between(dict_results["t"]/3600, dict_results["P_HE"]/1e6, 0, label=None, color=colors[2], alpha=0.25)
        plt.fill_between(dict_results["t"]/3600, dict_results["P_HP"]/1e6, 0, label=None, color=colors[1], alpha=0.25)

    plt.title("Power Distribution")
    plt.xlabel("Time (h)")  # Add x-axis label
    plt.ylabel("Power (MW)")  # Add y-axis label
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(loc='upper right')
    plt.xlim(0, max(dict_results["t"]/3600)) 
    plt.ylim(1.1*min(min(dict_results["P_HE"]/1e6), min(dict_results["P_HP"]/1e6)), 1.1*max(dict_results["P"]/1e6))  

    # Plot SoC
    plt.subplot(3, 2, 2)
    if "SOC_HE" in dict_results:
        plt.plot(dict_results["t_soc"]/3600, 100 * dict_results["SOC_HE"], label="HE Pack @ Beginning of Life", color=colors[2], linewidth=1.5)
        plt.plot(dict_results["t_soc"]/3600, 100 * dict_results["SOC_HE_aged"], label="HE Pack @ End of Life", linestyle='dashdot', color=colors[2], linewidth=1.5)
        plt.fill_between(dict_results["t_soc"]/3600, 100 * dict_results["SOC_HE"], 100 * dict_results["SOC_HE_aged"], label=None, color=colors[2], alpha=0.25)

        plt.plot(dict_results["t_soc"]/3600, 100 * dict_results["SOC_HP"], label="HP Pack @ Beginning of Life", color=colors[1], linewidth=1.5)
        plt.plot(dict_results["t_soc"]/3600, 100 * dict_results["SOC_HP_aged"], label="HP Pack @ End of Life", linestyle='dashdot', color=colors[1], linewidth=1.5)
        plt.fill_between(dict_results["t_soc"]/3600, 100 * dict_results["SOC_HP"], 100 * dict_results["SOC_HP_aged"], label=None, color=colors[1], alpha=0.25)

    else:
        plt.plot(dict_results["t_soc"]/3600, 100 * dict_results["SOC"], label="State of Charge @ Beginning of Life", color=colors[0])
        plt.plot(dict_results["t_soc"]/3600, 100 * dict_results["SOC_aged"], label="State of Charge @ End of Life", linestyle='dashdot')

    plt.title("State of Charge")
    plt.xlabel("Time (h)")  # Add x-axis label
    plt.ylabel("State of Charge (%)")  # Add y-axis label
    plt.ylim(0, 100)
    plt.xlim(0, max(dict_results["t"]/3600))  # Assuming time is always positive
    plt.axhline(y=10, color=plt.rcParams['grid.color'], alpha=plt.rcParams['grid.alpha'], linestyle='--', linewidth=1) 
    plt.axhline(y=90, color=plt.rcParams['grid.color'], alpha=plt.rcParams['grid.alpha'], linestyle='--', linewidth=1)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(loc='upper right')

    # Plot Voltage
    plt.subplot(3, 2, 3)
    if "V_HE" in dict_results:
        plt.plot(dict_results["t"]/3600, dict_results["V_HE"], label="High Energy Pack", color=colors[2])
        plt.plot(dict_results["t"]/3600, dict_results["V_HP"], label="High Power Pack", color=colors[1])
        plt.plot(dict_results["t"]/3600, dict_results["V_HE_aged"], label="High Energy Pack @ EOL", color=colors[2], linestyle='dashdot')
        plt.plot(dict_results["t"]/3600, dict_results["V_HP_aged"], label="High Power Pack @ EOL", color=colors[1], linestyle='dashdot')
    else:
        plt.plot(dict_results["t"]/3600, dict_results["V"], label="Voltage", color=colors[2])
        plt.plot(dict_results["t"]/3600, dict_results["V_aged"], label="Voltage @ EOL", color=colors[2], linestyle='dashdot')
    plt.title("Pack Voltage")
    plt.xlabel("Time (h)")  # Add x-axis label
    plt.ylabel("Voltage (V)")  # Add y-axis label
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(loc='upper right')
    plt.xlim(0, max(dict_results["t"]/3600))
    plt.ylim(0.98*min(min(dict_results["V_HE"]), min(dict_results["V_HP"])), 1.02*max(max(dict_results["V_HE"]), max(dict_results["V_HP"]))) 

    # Plot Current
    plt.subplot(3, 2, 4)
    if "I_HE" in dict_results:
        plt.plot(dict_results["t"]/3600, dict_results["I_HE"], label="High Energy Pack", color=colors[2])
        plt.plot(dict_results["t"]/3600, dict_results["I_HP"], label="High Power Pack", color=colors[1])
        plt.plot(dict_results["t"]/3600, dict_results["I_HE_aged"], label="High Energy Pack @ EOL", linestyle='dashdot', color=colors[2])
        plt.plot(dict_results["t"]/3600, dict_results["I_HP_aged"], label="High Power Pack @ EOL", linestyle='dashdot', color=colors[1])
        plt.axhline(y=dict_results["I_HE_rated"], color=plt.rcParams['grid.color'], alpha=plt.rcParams['grid.alpha'], linestyle='--', label=f"HE rated limit: {dict_results['I_HE_rated']:0.1f} A") 
        plt.axhline(y=dict_results["I_HP_rated"], color=plt.rcParams['grid.color'], alpha=plt.rcParams['grid.alpha'], linestyle='--', label=f"HP rated limit: {dict_results['I_HP_rated']:0.1f}A ") 

    else:
        plt.plot(dict_results["t"]/3600, dict_results["I"], label="Current", color=colors[2])
        plt.plot(dict_results["t"]/3600, dict_results["I_aged"], label="Current @ EOL", linestyle='dashdot')
        plt.axhline(y=dict_results["I_rated"], color=plt.rcParams['grid.color'], alpha=plt.rcParams['grid.alpha'], linestyle='--', label=f"limit: {dict_results['I_rated']:0.2f}") 
        
    
    plt.title("Pack Current")
    plt.xlabel("Time (h)")  # Add x-axis label
    plt.ylabel("Current (A)")  # Add y-axis label
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(loc='upper right')
    plt.xlim(0, max(dict_results["t"]/3600))
    plt.ylim(0, 1.1*max(max(dict_results["I_HE"]), max(dict_results["I_HP"])))

    # Plot Joule losses
    plt.subplot(3, 2, 5)
    if "P_HE_joule" in dict_results:
        plt.plot(dict_results["t"]/3600, dict_results["P_HE_joule"]/1000, label="High Energy Pack", color=colors[2])
        plt.plot(dict_results["t"]/3600, dict_results["P_HP_joule"]/1000, label="High Power Pack", color=colors[1])
    else:
        plt.plot(dict_results["t"]/3600, dict_results["P_joule"]/1000, label="Joule losses", color=colors[2])
    plt.title("Joule losses")
    plt.xlabel("Time (h)")  # Add x-axis label
    plt.ylabel("Power (kW)")  # Add y-axis label
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(loc='upper right')
    plt.xlim(0, max(dict_results["t"]/3600))
    plt.ylim(0, 1.1*max(max(dict_results["P_HE_joule"]/1000), max(dict_results["P_HP_joule"])/1000))

    # Plot aging (bar chart)
    plt.subplot(3, 2, 6)

    if "E_HE_aged" in dict_results:
        labels = ["HE Energy", "HE Energy @ EOL", "HP Energy", "HP Energy @ EOL"]
        values = [dict_results["E_HE"], dict_results["E_HE_aged"], dict_results["E_HP"], dict_results["E_HP_aged"]]
        bars = plt.bar(labels, values, color=[colors[2], colors[2], colors[1], colors[1]])
        plt.title("Battery Cell Degradation (Aging)")
        plt.xlabel("Battery Cell")
        plt.ylabel("Energy Content")
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.ylim(0, 1.2*max(dict_results["E_HE"], dict_results["E_HP"]))

        for bar, value in zip(bars, values):
            plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05, f"{value:0.2f} kWh", ha='center', va='bottom')
            
    else:
        labels = ["Energy", "Energy @ EOL"]
        values = [dict_results["E"], dict_results["E_aged"]]
        bars = plt.bar(labels, values, color=[colors[2], colors[2]])
        plt.title("Battery Cell Degradation (Aging)")
        plt.xlabel("Battery Cell")
        plt.ylabel("Energy Content")
        plt.grid(True, linestyle='--', alpha=0.7)

        for bar, value in zip(bars, values):
            plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05, f"{value:0.2f} kWh", ha='center', va='bottom')




    # Adjust layout to prevent clipping
    plt.tight_layout(pad=0.1)

    # Show the plots
    plt.show()

    return plt

def compare_power(dict_mono_HE, dict_mono_HP, dict_treshold, dict_opti, dict_opti2):
    gs = gridspec.GridSpec(4, 1)
    gs.update(wspace=0.5)


    ax1 = plt.subplot(gs[0, 0])
    plt.plot(dict_mono_HE["t"]/3600, dict_mono_HE["P"]/1000, label="Power Demand")
    plt.title("Monotype")
    plt.xlabel("Time (s)")  # Add x-axis label
    plt.ylabel("Power (kW)")  # Add y-axis label
    plt.grid(True)
    plt.legend()

    ax2 = plt.subplot(gs[1, 0])
    plt.plot(dict_treshold["t"]/3600, dict_treshold["P"]/1000, label="Power Demand")
    plt.plot(dict_treshold["t"]/3600, dict_treshold["P_HE"]/1000, label="High Energy Pack")
    plt.plot(dict_treshold["t"]/3600, dict_treshold["P_HP"]/1000, label="High Power Pack")
    plt.title("Rule-based minimization")
    plt.xlabel("Time (s)")  # Add x-axis label
    plt.ylabel("Power (kW)")  # Add y-axis label
    plt.grid(True)
    plt.legend()

    ax3 = plt.subplot(gs[2, 0])
    plt.plot(dict_opti["t"]/3600, dict_opti["P"]/1000, label="Power Demand")
    plt.plot(dict_opti["t"]/3600, dict_opti["P_HE"]/1000, label="High Energy Pack")
    plt.plot(dict_opti["t"]/3600, dict_opti["P_HP"]/1000, label="High Power Pack")
    plt.title("Nonlinear optimization")
    plt.xlabel("Time (s)")  # Add x-axis label
    plt.ylabel("Power (kW)")  # Add y-axis label
    plt.grid(True)
    plt.legend()

    ax4 = plt.subplot(gs[3, 0])
    plt.plot(dict_opti2["t"]/3600, dict_opti2["P"]/1000, label="Power Demand")
    plt.plot(dict_opti2["t"]/3600, dict_opti2["P_HE"]/1000, label="High Energy Pack")
    plt.plot(dict_opti2["t"]/3600, dict_opti2["P_HP"]/1000, label="High Power Pack")
    plt.title("Nonlinear optimization w/ intercharging")
    plt.xlabel("Time (s)")  # Add x-axis label
    plt.ylabel("Power (kW)")  # Add y-axis label
    plt.grid(True)
    plt.legend()

    plt.show()

    return None

def compare_soc(dict_mono_HE, dict_mono_HP):
    plt.subplot(2, 1, 1)
    plt.plot(dict_mono_HE["t_soc"]/3600, 100 * dict_mono_HE["SOC"], label="SOC HE")
    plt.title("Pack State of Charge")
    plt.xlabel("Time (h)")  # Add x-axis label
    plt.ylabel("State of Charge (%)")  # Add y-axis label
    plt.ylim(0, 100)
    plt.axhline(y=10, color=plt.rcParams['grid.color'], alpha=plt.rcParams['grid.alpha'], linestyle='--') 
    plt.axhline(y=90, color=plt.rcParams['grid.color'], alpha=plt.rcParams['grid.alpha'], linestyle='--')
    plt.grid(True)
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(dict_mono_HP["t_soc"]/3600, 100 * dict_mono_HP["SOC"], label="SOC HP")
    plt.title("Pack State of Charge")
    plt.xlabel("Time (h)")  # Add x-axis label
    plt.ylabel("State of Charge (%)")  # Add y-axis label
    plt.ylim(0, 100)
    plt.axhline(y=10, color=plt.rcParams['grid.color'], alpha=plt.rcParams['grid.alpha'], linestyle='--') 
    plt.axhline(y=90, color=plt.rcParams['grid.color'], alpha=plt.rcParams['grid.alpha'], linestyle='--')
    plt.grid(True)
    plt.legend()

    plt.tight_layout(pad=0.1)
    plt.show()

    return None

def plot_power(dict_results):
    colors = sns.color_palette("Set2", 3)
    plt.figure(figsize=(12, 6)) #8 by 3.2

    plt.plot(dict_results["t"]/3600, dict_results["P"]/1e6, label="Power Demand", color=colors[0] if "P_HE" in dict_results else colors[1], linewidth=1.5)
    plt.fill_between(dict_results["t"]/3600, dict_results["P"]/1e6, 0, label=None, color=colors[0], alpha=0.35)

    if "P_HE" in dict_results:
        plt.plot(dict_results["t"]/3600, dict_results["P_HE"]/1e6, label="High Energy Pack", color=colors[2], linewidth=1.5)
        plt.plot(dict_results["t"]/3600, dict_results["P_HP"]/1e6, label="High Power Pack", color=colors[1], linewidth=1.5)

        plt.fill_between(dict_results["t"]/3600, dict_results["P_HE"]/1e6, 0, label=None, color=colors[2], alpha=0.25)
        plt.fill_between(dict_results["t"]/3600, dict_results["P_HP"]/1e6, 0, label=None, color=colors[1], alpha=0.25)

    plt.title("Power Distribution")
    plt.xlabel("Time (h)")  # Add x-axis label
    plt.ylabel("Power (MW)")  # Add y-axis label
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(loc='upper right')
    plt.xlim(0, max(dict_results["t"]/3600))  # Assuming time is always positive
    #plt.ylim(min(dict_results["P"]/1e6), max(dict_results["P"]/1e6 + 0.2, default=10))  # Set a default upper limit for y-axis if needed
    plt.show()

    return None

def plot_soc(dict_results):
    colors = sns.color_palette("Set2", 3)
    plt.figure(figsize=(12, 6))


    if "SOC_HE" in dict_results:
        plt.plot(dict_results["t_soc"]/3600, 100 * dict_results["SOC_HE"], label="HE Pack @ Beginning of Life", color=colors[2], linewidth=1.5)
        plt.plot(dict_results["t_soc"]/3600, 100 * dict_results["SOC_HE_aged"], label="HE Pack @ End of Life", linestyle='dashdot', color=colors[2], linewidth=1.5)
        plt.fill_between(dict_results["t_soc"]/3600, 100 * dict_results["SOC_HE"], 100 * dict_results["SOC_HE_aged"], label=None, color=colors[2], alpha=0.25)

        plt.plot(dict_results["t_soc"]/3600, 100 * dict_results["SOC_HP"], label="HP Pack @ Beginning of Life", color=colors[1], linewidth=1.5)
        plt.plot(dict_results["t_soc"]/3600, 100 * dict_results["SOC_HP_aged"], label="HP Pack @ End of Life", linestyle='dashdot', color=colors[1], linewidth=1.5)
        plt.fill_between(dict_results["t_soc"]/3600, 100 * dict_results["SOC_HP"], 100 * dict_results["SOC_HP_aged"], label=None, color=colors[1], alpha=0.25)

    else:
        plt.plot(dict_results["t_soc"]/3600, 100 * dict_results["SOC"], label="State of Charge @ Beginning of Life", color=colors[0])
        plt.plot(dict_results["t_soc"]/3600, 100 * dict_results["SOC_aged"], label="State of Charge @ End of Life", linestyle='dashdot')

    plt.title("State of Charge")
    plt.xlabel("Time (h)")  # Add x-axis label
    plt.ylabel("State of Charge (%)")  # Add y-axis label
    plt.ylim(0, 100)
    plt.xlim(0, max(dict_results["t"]/3600))  # Assuming time is always positive
    plt.axhline(y=10, color=plt.rcParams['grid.color'], alpha=plt.rcParams['grid.alpha'], linestyle='--', linewidth=1) 
    plt.axhline(y=90, color=plt.rcParams['grid.color'], alpha=plt.rcParams['grid.alpha'], linestyle='--', linewidth=1)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(loc='upper right')
    plt.show()

    for color in colors:
        print(color)

    return None



def plot_inputs(loads):
    colors = sns.color_palette("Set2", 3)

    for i in range(len(loads)):
        plt.subplot(len(loads), 1, i+1)
        plt.plot(loads[i]["t"]/3600, loads[i]["P"]/1e6, label="Power Demand", color=colors[0], linewidth=1.5)
        plt.fill_between(loads[i]["t"]/3600, loads[i]["P"]/1e6, 0, label=None, color=colors[0], alpha=0.35)

        plt.title(f"Power Demand #{i+1}")
        plt.xlabel("Time (h)")  # Add x-axis label
        plt.ylabel("Power (MW)")  # Add y-axis label
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend(loc='upper right')
        plt.xlim(0, max(loads[i]["t"]/3600))
        plt.ylim(1.1*min(0, min(loads[i]["P"]/1e6)), 1.1*max(loads[i]["P"]/1e6)) 

    plt.tight_layout()
    plt.show()

    return None



def plot_multiple(dict_results):
    colors = sns.color_palette("Set2", 3)

    for i in range(len(dict_results["t"])):
        # Power split
        plt.subplot(4, len(dict_results["t"]), i+1)

        plt.plot(dict_results["t"][i]/3600, dict_results["P"][i]/1e6, label="Power Demand", color=colors[0], linewidth=1.5)
        plt.fill_between(dict_results["t"][i]/3600, dict_results["P"][i]/1e6, 0, label=None, color=colors[0], alpha=0.35)

        plt.title(f"Power Split #{i+1}")
        plt.xlabel("Time (h)")  # Add x-axis label
        plt.ylabel("Power (MW)")  # Add y-axis label
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend(loc='upper right')
        plt.xlim(0, max(dict_results["t"][i]/3600))
        plt.ylim(1.1*min(0, min(dict_results["P"][i]/1e6)), 1.1*max(dict_results["P"][i]/1e6))

        # State of Charge
        plt.subplot(4, len(dict_results["t"]), i+1+len(dict_results["t"]))

        plt.plot(dict_results["t_soc"][i]/3600, dict_results["SOC"][i]*1e2, label="SoC @ BOL", color=colors[0], linewidth=1.5)
        plt.plot(dict_results["t_soc"][i]/3600, dict_results["SOC_aged"][i]*1e2, label="SoC @ EOL", color=colors[0], linewidth=1.5, linestyle='dashdot')
        plt.fill_between(dict_results["t_soc"][i]/3600, dict_results["SOC"][i]*1e2, dict_results["SOC_aged"][i]*1e2, label=None, color=colors[0], alpha=0.35)

        plt.axhline(y=10, color=plt.rcParams['grid.color'], alpha=plt.rcParams['grid.alpha'], linestyle='--', linewidth=1) 
        plt.axhline(y=90, color=plt.rcParams['grid.color'], alpha=plt.rcParams['grid.alpha'], linestyle='--', linewidth=1)

        plt.title(f"State of Charge #{i+1}")
        plt.xlabel("Time (h)")  # Add x-axis label
        plt.ylabel("State of Charge (%)")  # Add y-axis label
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend(loc='upper right')
        plt.xlim(0, max(dict_results["t"][i]/3600))
        plt.ylim(0, 100)


        # Voltage
        plt.subplot(4, len(dict_results["t"]), i+1+2*len(dict_results["t"]))

        plt.plot(dict_results["t"][i]/3600, dict_results["V"][i], label="Voltage @ BOL", color=colors[0], linewidth=1.5)
        plt.plot(dict_results["t"][i]/3600, dict_results["V_aged"][i], label="Voltage @ EOL", color=colors[0], linewidth=1.5, linestyle='dashdot')
        plt.fill_between(dict_results["t"][i]/3600, dict_results["V"][i], dict_results["V_aged"][i], label=None, color=colors[0], alpha=0.35)

        plt.title(f"Voltage #{i+1}")
        plt.xlabel("Time (h)")  # Add x-axis label
        plt.ylabel("Voltage (V)")  # Add y-axis label
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend(loc='upper right')
        plt.xlim(0, max(dict_results["t"][i]/3600))
        plt.ylim(0.95*min(dict_results["V"][i]), 1.05*max(dict_results["V"][i]))

        # Current
        plt.subplot(4, len(dict_results["t"]), i+1+3*len(dict_results["t"]))

        plt.plot(dict_results["t"][i]/3600, dict_results["I"][i], label="Current @ BOL", color=colors[0], linewidth=1.5)
        plt.plot(dict_results["t"][i]/3600, dict_results["I_aged"][i], label="Current @ EOL", color=colors[0], linewidth=1.5, linestyle='dashdot')
        plt.fill_between(dict_results["t"][i]/3600, dict_results["I"][i], dict_results["I_aged"][i], label=None, color=colors[0], alpha=0.35)
        plt.axhline(y=dict_results["I_rated"], color=plt.rcParams['grid.color'], alpha=plt.rcParams['grid.alpha'], linestyle='--', label=f"limit: {dict_results['I_rated']:0.2f} A")

        plt.title(f"Current #{i+1}")
        plt.xlabel("Time (h)")  # Add x-axis label
        plt.ylabel("Current (A)")  # Add y-axis label
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend(loc='upper right')
        plt.xlim(0, max(dict_results["t"][i]/3600))
        plt.ylim(1.1*min(0, min(dict_results["I"][i])), 1.1*max(dict_results["I"][i]))


    plt.tight_layout(pad=0)
    plt.show()
        
    return None
