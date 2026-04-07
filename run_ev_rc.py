import streamlit as st
import numpy as np

st.set_page_config(page_title="Online Reference Crop Evaporation Rate Calculator")

def calculate_erc(windspeed, albedo, n, lat, latmin,
                  elevation, Tmax, Tmin, rhum, J):

    P          = 101.3 * ((293 - 0.0065 * elevation) / 293) ** 5.256
    Stefan     = 4.903e-9
    Tmean      = (Tmax + Tmin) / 2.0
    lam        = 2.501 - 0.002361 * Tmean

    e_tmax     = 0.6108 * np.exp(17.27 * Tmax / (237.3 + Tmax))
    e_tmin     = 0.6108 * np.exp(17.27 * Tmin / (237.3 + Tmin))
    e_sat      = (e_tmax + e_tmin) / 2.0
    e_act      = e_sat * rhum

    gamma      = 0.0016286 * P / lam
    D          = e_sat - e_act
    Delta      = 4098 * e_sat / (237.3 + Tmean) ** 2
    denom      = Delta + gamma * (1 + 0.34 * windspeed)  # FAO-56

    delta_sun  = 0.4093 * np.sin(2 * np.pi * J / 365 - 1.405)
    phi        = np.pi / 180 * (lat + latmin / 60.0)
    ws         = np.arccos(np.clip(-np.tan(phi) * np.tan(delta_sun), -1.0, 1.0))
    N          = 24 * ws / np.pi
    dr         = 1 + 0.033 * np.cos(2 * np.pi * J / 365)
    Isd        = 15.392 * dr * (ws * np.sin(phi) * np.sin(delta_sun)
                               + np.cos(phi) * np.cos(delta_sun) * np.sin(ws))

    Iscd       = (0.25 + 0.5 * n / N) * Isd if N > 0 else 0.0
    Sn         = Iscd * (1 - albedo)                          # MJ/m²/day

    E_emis     = 0.34 - 0.14 * np.sqrt(max(e_act, 0))
    f_cloud    = Iscd / Isd if Isd > 0 else 0.0
    Ln         = -f_cloud * E_emis * Stefan * (Tmean + 273.15) ** 4  # MJ/m²/day

    Rnet       = Sn + Ln                                      # MJ/m²/day

    Erc = ( (Delta / denom) * (Rnet / lam)                   # Rnet mm/day'e çevriliyor
          + (gamma  / denom) * (900 / (Tmean + 273)) * windspeed * D )

    return round(Erc, 2)

st.title("Online Reference Crop Evaporation Rate Calculator")

# User input
windspeed = st.number_input("Windspeed (m/s)", value=3.0)
albedo = st.number_input("Albedo", value=0.23)
n = st.number_input("Hours of Bright Sunshine", value=6.0)
lat = st.number_input("Latitude", value=30.0)
latmin = st.number_input("Latitude Minutes", value=0.0)
elevation = st.number_input("Elevation (m)", value=1000.0)
Tmax = st.number_input("Maximum Temperature (°C)", value=15.0)
Tmin = st.number_input("Minimum Temperature (°C)", value=5.0)
rhum = st.number_input("Relative Humidity (Decimal, e.g., 0.80)", value=0.80) # FIXED: Updated label for clarity
J = st.number_input("Julian Day", value=105.0)

# Calculate Erc
Erc = calculate_erc(windspeed, albedo, n, lat, latmin, elevation, Tmax, Tmin, rhum, J)

# Added a nice Streamlit metric container for a cleaner look
st.subheader("Results")
st.metric(label="Reference Crop Evaporation (Erc)", value=f"{Erc} mm/day")
