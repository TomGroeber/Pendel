import streamlit as st
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

def foucault_equations(y, t, omega, g, l):
    drdt = np.zeros(6)
    drdt[0:3] = y[3:6]
    drdt[3:6] = -g/l * np.array([y[0], y[1], 0]) - 2 * np.cross(omega, y[3:6])
    return drdt

def foucault_angular_velocity(omega_0, latitude):
    theta = np.deg2rad(latitude)
    omega_pendulum = omega_0 * np.array([np.cos(theta), 0, np.sin(theta)])
    return omega_pendulum

def solve_pendulum(latitude):
    g = 9.81
    l = 2

    t_span = np.linspace(0, 3600, 1000)
    omega_0 = 2 * np.pi / (24 * 3600)
    initial_conditions = [1, 0, 0, 1, 0, 0]

    omega = foucault_angular_velocity(omega_0, latitude)
    solution = odeint(foucault_equations, initial_conditions, t_span, args=(omega, g, l))

    return t_span, solution

# Streamlit-App
st.title('Foucaultsches Pendel Simulation')

locations = {
    'Nordpol': 90,
    'Innsbruck': 47.2682,
    'Barcelona': 41.3851,
    'London': 51.5099,
    'New York': 40.7128,
    'Tokio': 35.6895,
    'Kenia': -1.2921,
    'Kapstadt': -33.918861,  # Koordinaten für Kapstadt
}

selected_location = st.selectbox('Wähle eine Location:', list(locations.keys()))

latitude = locations[selected_location]
t, solution = solve_pendulum(latitude)

# Plot
fig, ax = plt.subplots(figsize=(8, 8))
ax.plot(solution[:, 0], solution[:, 1], label=selected_location)
ax.set_xlabel('X-Koordinate')
ax.set_ylabel('Y-Koordinate')
ax.set_title(f'Pendelbewegung in {selected_location} für 1 Stunde')
ax.legend()
ax.set_xlim([-1.2, 1.2])
ax.set_ylim([-1.2, 1.2])
ax.grid(True)

st.pyplot(fig)
