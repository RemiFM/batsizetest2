import pandas as pd
import altair as alt

hex_color = ["#66c298", "#8da0cb", "#fc8d62"]

def plot_load_profiles(profiles, height):
    # Scale inputs
    chart_data = profiles[0].copy()       
    chart_data.loc[:, 't'] = chart_data.loc[:, 't'] / 3600            #seconds to hours
    chart_data.loc[:, 'P'] = chart_data.loc[:, 'P'] / 1000          #watts to kilowatts

    # Restructure DataFrame
    chart_data = (chart_data
                  .loc[:, ['t', 'P']]
                  .rename(columns = {'P':'Power Demand 1'})
                  .melt('t'))

    # Create chart object
    chart_1 = (
        alt.Chart(data = chart_data)
        .mark_area(interpolate='linear', fillOpacity=0.3)
        .encode(
        x = alt.X('t', axis = alt.Axis(title = 'Time (h)', grid = True)),
        y = alt.Y('value', axis = alt.Axis(title = 'Power (kW)')),
        color = alt.Color('variable', scale=alt.Scale(range=hex_color), sort = ['Power Demand 1'], legend = alt.Legend(orient = 'top', title = 'None', titleOpacity = 0, titlePadding = 0, titleFontSize = 0)) 
        )
        .properties(
            height = height,
        )
        .interactive()
    )

    # Create a separate line chart for the outline
    chart_1_outline = (
        alt.Chart(data=chart_data)
        .mark_line(interpolate='linear', strokeWidth=1)  # Set the desired stroke width
        .encode(
            x=alt.X('t', axis=alt.Axis(title='Time (h)', grid=True)),
            y=alt.Y('value'),
            color=alt.Color('variable', scale=alt.Scale(range=hex_color)),
        )
    )

    chart = chart_1 + chart_1_outline

    if len(profiles) >= 2:
        # Scale inputs
        chart_data = profiles[1].copy()                                     #copy of DataFrame for scaling
        chart_data.loc[:, 't'] = chart_data.loc[:, 't'] / 3600            #seconds to hours
        chart_data.loc[:, 'P'] = chart_data.loc[:, 'P'] / 1000          #watts to kilowatts

        # Restructure DataFrame
        chart_data = (chart_data
                    .loc[:, ['t', 'P']]
                    .rename(columns = {'P':'Power Demand 2'})
                    .melt('t'))

        # Create chart object
        chart_2 = (
            alt.Chart(data = chart_data)
            .mark_area(interpolate='linear', fillOpacity=0.3)
            .encode(
            x = alt.X('t', axis = alt.Axis(title = 'Time (h)', grid = True)),
            y = alt.Y('value', axis = alt.Axis(title = 'Power (kW)')),
            color = alt.Color('variable', scale=alt.Scale(range=hex_color), sort = ['Power Demand 2'], legend = alt.Legend(orient = 'top', title = 'None', titleOpacity = 0, titlePadding = 0, titleFontSize = 0)) 
            )
            .properties(
                height = height,
            )
            .interactive()
        )

        # Create a separate line chart for the outline
        chart_2_outline = (
            alt.Chart(data=chart_data)
            .mark_line(interpolate='linear', strokeWidth=1)  # Set the desired stroke width
            .encode(
                x=alt.X('t', axis=alt.Axis(title='Time (h)', grid=True)),
                y=alt.Y('value'),
                color=alt.Color('variable', scale=alt.Scale(range=hex_color)),
            )
        )

        chart = chart + chart_2 + chart_2_outline

    return chart




def plot_powers(dict_result, title="Power Split", height=380):
    # Scale inputs
    chart_data = pd.DataFrame({
        't': dict_result['t'],
        'P': dict_result['P'],
        'P_HE': dict_result['P_HE'],
        'P_HP': dict_result['P_HP']
        })
                                 #copy of DataFrame for scaling
    chart_data.loc[:, 't'] = chart_data.loc[:, 't'] / 3600            #seconds to hours
    chart_data.loc[:, 'P'] = chart_data.loc[:, 'P'] / 1000          #watts to kilowatts
    chart_data.loc[:, 'P_HE'] = chart_data.loc[:, 'P_HE'] / 1000          #watts to kilowatts
    chart_data.loc[:, 'P_HP'] = chart_data.loc[:, 'P_HP'] / 1000          #watts to kilowatts

    # Restructure DataFrame
    chart_data = (chart_data
                  .loc[:, ['t', 'P', 'P_HE', 'P_HP']]
                  .rename(columns = {'P':'Total power demand', 'P_HE':'Power from HE', 'P_HP':'Power from HP'})
                  .melt('t'))

    # Create chart object
    chart = (
        alt.Chart(data=chart_data)
        .mark_area(interpolate='linear', fillOpacity=0.3)
        .encode(
            x=alt.X('t', axis=alt.Axis(title='Time (min)', grid=True)),
            y=alt.Y('value', axis=alt.Axis(title='Power (kW)')),
            color=alt.Color('variable:N', sort=['Total power demand'], scale=alt.Scale(range=hex_color),  legend=alt.Legend(orient='bottom', title='None', titleOpacity=0, titlePadding=0, titleFontSize=0)),
        )
        .properties(
            #height=height,
            title={
                "text": title,
                "anchor": "middle"
            }
        )
        .interactive()
    )

    # Create a separate line chart for the outline
    outline_chart = (
        alt.Chart(data=chart_data)
        .mark_line(interpolate='linear', strokeWidth=1)  # Set the desired stroke width
        .encode(
            x=alt.X('t', axis=alt.Axis(title='Time (min)', grid=True)),
            y=alt.Y('value'),
            color=alt.Color('variable:N', scale=alt.Scale(range=hex_color), sort=['Total power demand']),
        )
    )

    # Combine the area chart and the outline chart
    combined_chart = chart + outline_chart

    return combined_chart




def plot_SOC(dict_result, height=320, title="State of Charge"):
    hex_color = ["#66c298", "#fc8d62", "#8da0cb"]

    # Scale inputs
    chart_data = pd.DataFrame({
        't': dict_result['t_soc'],
        'SOC_HE': dict_result['SOC_HE'] * 100,  # Scale to percentage
        'SOC_HP': dict_result['SOC_HP'] * 100,  # Scale to percentage
        'SOC_HE_aged': dict_result['SOC_HE_aged'] * 100,  # Scale to percentage
        'SOC_HP_aged': dict_result['SOC_HP_aged'] * 100  # Scale to percentage
    })

    # Restructure DataFrame
    chart_data = (
        chart_data
        .loc[:, ['t', 'SOC_HE', 'SOC_HP', 'SOC_HE_aged', 'SOC_HP_aged']]
        .rename(columns={'SOC_HE': 'SoC of HE', 'SOC_HP': 'SoC of HP', 'SOC_HE_aged': 'SoC of HE (Aged)', 'SOC_HP_aged': 'SoC of HP (Aged)'})
        .melt('t')
    )

    # Create chart object
    chart = (
        alt.Chart(data=chart_data)
        .mark_line(interpolate='linear')
        .encode(
            x=alt.X('t', axis=alt.Axis(title='Time (min)', grid=True)),
            y=alt.Y('value', axis=alt.Axis(title='State of Charge (%)'), scale=alt.Scale(domain=[0, 100])),
            color=alt.Color('variable:N', scale=alt.Scale(range=hex_color[1:]), sort=['SoC of HE'], legend=alt.Legend(orient='bottom', title='', titleOpacity=0, titlePadding=0, titleFontSize=0)),
            strokeDash=alt.StrokeDash('variable:N', scale=alt.Scale(domain=['SoC of HE', 'SoC of HE (Aged)', 'SoC of HP', 'SoC of HP (Aged)'], range=[[1, 0], [4, 2], [1, 0], [4, 2]]))  # Dotted lines for aged variables
        )
        .properties(
            height=height,
            title={
                "text": title,
                "anchor": "middle"
            }
        )
        .configure_legend(orient="bottom") 
        .interactive()
    )
    return chart

# Example usage:
# dict_result = {'t_soc': [0, 1, 2, 3, 4], 'SOC_HE': [10, 20, 30, 40, 50], 'SOC_HP': [5, 15, 25, 35, 45], 'SOC_HE_aged': [8, 18, 28, 38, 48], 'SOC_HP_aged': [3, 8, 18, 28, 38]}
# plot_SOC(dict_result).show()
