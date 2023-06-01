import streamlit as st
import numpy as np
import streamlit as st
from datetime import datetime

st.set_page_config(page_title="Online Reference Crop Evaporation Rate Calculator")

def calculate_erc(windspeed, albedo, n, lat, latmin, elevation, Tmax, Tmin, rhum, J):
    P=101.3*((293-0.0065*elevation)/293)**5.256 #kPa
    Stefboltzcons = 4.903 * 10**-9 #MJ K-4 m-2 day-1
    Tmean = Tmax / 2 + Tmin / 2
    λ = 2.501 - 0.0002361*Tmean # latent heat of vaporization λ  [MJ/kg]
    e_tmax = 0.6108 * np.exp((17.27*Tmax)/(237.3+Tmax)) # kPa
    e_tmin = 0.6108 * np.exp((17.27*Tmin)/(237.3+Tmin)) # kPa
    e_sat = (e_tmax/2 + e_tmin/2) * rhum # kPa
    γ = 0.0016286*P/λ # psychrometric constant γ [kPa/°C]
    D = (e_tmax/2 + e_tmin/2) * (100-rhum*100)/100# Vapor pressure deficit kPa
    Δ = 4098*e_sat/((237.3+Tmean) **2) # kPa/C  The gradient of the relationship between saturated vapor pressure and temperature is often used in equations describing evaporation rate and, when used in this way, this gradient is usually represented by Δ
    γmod = γ * (1 + 0.33 * windspeed) # modified psychrometric constant
    δ = 0.4093 * np.sin(2*np.pi*J/365-1.405) # and the solar declination, δ, are given by
    φ = np.pi/180 * (lat + latmin/60) # The latitude, φ, expressed in radians is positive for the northern hemisphere and negative for the southern hemisphere. The conversion from decimal degrees to radians is given by:
    ws  = np.arccos(-np.tan(φ) * np.tan(δ)) #the sunset hour angle 
    N = 24 * ws/np.pi # The daylight hours, N
    dr = 1 + 0.033 * np.cos(2*np.pi*J/365) # The inverse relative distance Earth-Sun, dr
    Isd = 15.392 * dr * (ws*np.sin(φ) * np.sin(δ) + np.cos(φ)*np.cos(δ)*np.sin(ws)) # Extraterrestial Solar Radiation mm/day
    Iscd = (0.25 + 0.5*n/N) * Isd # solar radiation at the ground mm/day
    Sn = Iscd * (1-albedo) # Net solar radiation mm/day
    E = 0.34 - 0.14*(e_sat**0.5) # Emissivity 
    f = Iscd / Isd # Cloud factor for Longwave Radiation
    Ln = -f * E * Stefboltzcons * ((Tmean+273.15)**4) / λ
    Rnet = Sn + Ln
    G = 0 # mm/day neglected soil heat flux
    Erc = (Δ/(Δ+γmod))*(Rnet-G)+((γ/(Δ+γmod))*(900/(Tmean + 275))) * windspeed * D # reference crop evaporation
    return round(Erc, 2)

st.title("Online Reference Crop Evaporation Rate Calculator")

# User input
windspeed = st.number_input("Windspeed (m/s)", value=3)
albedo = st.number_input("Albedo", value=0.23)
n = st.number_input("Hours of Bright Sunshine", value=6)
lat = st.number_input("Latitude", value=30)
latmin = st.number_input("Latitude Minutes", value=0)
elevation = st.number_input("Elevation (m)", value=1000)
Tmax = st.number_input("Maximum Temperature (°C)", value=15)
Tmin = st.number_input("Minimum Temperature (°C)", value=5)
rhum = st.number_input("Relative Humidity (%)", value=0.80)
J = st.number_input("Julian Day", value=105)


# Calculate Erc
Erc = calculate_erc(windspeed, albedo, n, lat, latmin, elevation, Tmax, Tmin, rhum, J)
st.write("Reference Crop Evaporation is ", Erc)


