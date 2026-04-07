import streamlit as st
import numpy as np

st.set_page_config(page_title="Online Reference Crop Evaporation Rate Calculator")

def calculate_erc(windspeed, albedo, n, lat, latmin, elevation, Tmax, Tmin, rhum, J):
    P=101.3*((293-0.0065*elevation)/293)**5.256 #kPa
    Stefboltzcons = 4.903 * 10**-9 #MJ K-4 m-2 day-1
    Tmean = Tmax / 2 + Tmin / 2
    λ = 2.501 - 0.002361*Tmean # latent heat of vaporization λ  [MJ/kg]
    
    e_tmax = 0.6108 * np.exp((17.27*Tmax)/(237.3+Tmax)) # kPa
    e_tmin = 0.6108 * np.exp((17.27*Tmin)/(237.3+Tmin)) # kPa
    
    # FIXED: Split saturated and actual vapor pressure
    e_sat = (e_tmax/2 + e_tmin/2) # kPa (Saturated)
    e_act = e_sat * rhum # kPa (Actual)
    
    γ = 0.0016286*P/λ # psychrometric constant γ [kPa/°C]
    D = e_sat - e_act # FIXED: Simplified Vapor pressure deficit
    Δ = 4098*e_sat/((237.3+Tmean) **2) # FIXED: Now correctly uses true e_sat
    γmod = γ * (1 + 0.33 * windspeed) # modified psychrometric constant
    δ = 0.4093 * np.sin(2*np.pi*J/365-1.405) # and the solar declination, δ, are given by
    φ = np.pi/180 * (lat + latmin/60) # The latitude, φ, expressed in radians is positive for the northern hemisphere and negative for the southern hemisphere. The conversion from decimal degrees to radians is given by:
    ws  = np.arccos(-np.tan(φ) * np.tan(δ)) #the sunset hour angle 
    N = 24 * ws/np.pi # The daylight hours, N
    dr = 1 + 0.033 * np.cos(2*np.pi*J/365) # The inverse relative distance Earth-Sun, dr
    Isd = 15.392 * dr * (ws*np.sin(φ) * np.sin(δ) + np.cos(φ)*np.cos(δ)*np.sin(ws)) # Extraterrestial Solar Radiation mm/day
    Iscd = (0.25 + 0.5*n/N) * Isd # solar radiation at the ground mm/day
    Sn = Iscd * (1-albedo) # Net solar radiation mm/day
    E = 0.34 - 0.14*(e_act**0.5) # FIXED: Emissivity now correctly uses e_act
    f = Iscd / Isd # Cloud factor for Longwave Radiation
    Ln = -f * E * Stefboltzcons * ((Tmean+273.15)**4) / λ
    Rnet = Sn + Ln
    G = 0 # mm/day neglected soil heat flux
    
    #Reference Crop Evaporation
    Erc = (Δ/(Δ+γmod))*(Rnet-G)+((γ/(Δ+γmod))*(900/(Tmean + 275))) * windspeed * D # reference crop evaporation
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
