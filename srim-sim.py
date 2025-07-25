import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from tqdm import tqdm
import os
import shutil
import subprocess
from pathlib import Path
from itertools import product

# Define available foil thicknesses and maximum number of each type
dx = [25, 50, 125, 200, 300]  # µm - foil thicknesses available
dx_err = 5  # µm

n = 4  # maximum number of each foil type

# Generate all possible thickness combinations using itertools.product
thickness_set = set()
for combination in product(range(n), repeat=len(dx)):
    total_thickness = sum(count * thickness for count, thickness in zip(combination, dx))
    thickness_set.add(total_thickness)

tot_dx = sorted(list(thickness_set))

print("Possible Al combination (in µm):", tot_dx)

sub_set = tot_dx[1:3] # remove the first element for obvious reasons

print("Selected thicknesses for simulation (in µm):", sub_set)

E = 18.12  # MeV, beam energy -> from alex's beam energy measurement paper let's gooo!
SIGMA_E = 0.07  # MeV

N_PARTICLES = 1000

# srim asks about data saving and stuff before it exits, which we dont care about.
# We just force close it after the timeout expires
# make sure to set this timeout to a value that is enough for the simulation to complete
# or to simulate a sufficient number of particles
simulation_timeout = 20 # seconds

trim_simulation_directory = "TRIM_simulations"
os.makedirs(trim_simulation_directory, exist_ok=True)
TRIM_PATH = Path("/home/lars/Applications/SRIM-2013/TRIM.exe")

def run_trim_simulation(folder):
    shutil.copy(os.path.join(folder, "TRIM.IN"), TRIM_PATH.parent) # copy input file to trim dir
    cwd = os.getcwd()
    # ruin TRIM in background with timeout
    try:
        os.chdir(TRIM_PATH.parent) # Change to the directory where TRIM is located
        process = subprocess.Popen(f'wine {TRIM_PATH}', shell=True) # Use wine to run TRIM since, damn Windows
        process.wait(timeout=simulation_timeout)  # Wait max 10 seconds
    except subprocess.TimeoutExpired:
        process.terminate()  # Terminate if still running after timeout
        try:
            process.wait(timeout=2)  # Give it 2 seconds to terminate gracefully
        except subprocess.TimeoutExpired:
            process.kill()  # Force kill if it doesn't terminate
    finally:
        os.chdir(cwd)  # always change back to the original directory

    # copy results
    transmit_path = TRIM_PATH.parent / "SRIM Outputs" / "TRANSMIT.txt"
    if transmit_path.exists():
        destination_path = os.path.join(folder, "TRANSMIT.txt")
        os.makedirs(os.path.dirname(destination_path), exist_ok=True)
        shutil.copy(transmit_path, destination_path)


def write_input_file(tot_dx_cov, folder, energy, n_particles):
    thickness_angstrom = tot_dx_cov * 1e4
    E_keV = energy * 1000
    autosave_number = 10000  # AutoSave Number
    input_text_new = f"""==> SRIM-2013.00 This file controls TRIM Calculations.
Ion: Z1 ,  M1,  Energy (keV), Angle,Number,Bragg Corr,AutoSave Number.
     1   1.008       {E_keV:.0f}       0     {n_particles}         1    {autosave_number}
Cascades(1=No;2=Full;3=Sputt;4-5=Ions;6-7=Neutrons), Random Number Seed, Reminders
                      1                                   0       0
Diskfiles (0=no,1=yes): Ranges, Backscatt, Transmit, Sputtered, Collisions(1=Ion;2=Ion+Recoils), Special EXYZ.txt file
                          0       0           1       0               0                               0
Target material : Number of Elements & Layers
"H ({E_keV:.1f}) into Layer 1                     "       1               1
PlotType (0-5); Plot Depths: Xmin, Xmax(Ang.) [=0 0 for Viewing Full Target]
       0                         0         {int(thickness_angstrom)}
Target Elements:    Z   Mass(amu)
Atom 1 = Al =       13  26.982
Layer   Layer Name /               Width Density    Al(13)
Numb.   Description                (Ang) (g/cm3)    Stoich
 1      "Layer 1"           {int(thickness_angstrom)}  2.702       1
0  Target layer phases (0=Solid, 1=Gas)
0 
Target Compound Corrections (Bragg)
 1  
Individual target atom displacement energies (eV)
      25
Individual target atom lattice binding energies (eV)
       3
Individual target atom surface binding energies (eV)
    3.36
Stopping Power Version (1=2011, 0=2011)
 0 
"""
    # Write with Windows line endings (CRLF) since SRIM expects Windows format, damn Windows!!!!
    with open(os.path.join(folder, "TRIM.IN"), "w", newline='\r\n', encoding="utf-8") as f:
        f.write(input_text_new)

def extract_transmitted_energy(folder):
    filepath = os.path.join(folder, "TRANSMIT.txt")
    if not os.path.exists(filepath):
        return np.nan, np.nan, np.nan
    energies = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            if line.startswith("T"):
                parts = line.split()
                try:
                    energy_ev = float(parts[3])  # energy is on 4th column in units of eV
                    energy_mev = energy_ev / 1e6
                    energies.append(energy_mev)
                except (IndexError, ValueError):
                    continue
    if energies:
        energies = np.array(energies)
        mean = np.mean(energies)
        stde = np.std(energies, ddof=1)
        sem = stde / np.sqrt(len(energies))
        return mean, stde, sem
    return np.nan, np.nan, np.nan

estimated_duration = simulation_timeout * len(sub_set) * 3  # 3 runs per thickness (min, max, nominal)
print(f"Estimated total simulation duration: {estimated_duration} seconds ({estimated_duration / 60:.2f} minutes)")

results = []
for thickness in tqdm(sub_set, desc="Running simulations", unit="thickness"):
    print(f"Simulation for thickness: {thickness} µm")
    folder_base = os.path.join(trim_simulation_directory, f"{thickness}um")
    os.makedirs(folder_base, exist_ok=True)

    print(f"running simulation for thickness {thickness} µm in folder {folder_base}")
    write_input_file(thickness, folder_base, E, N_PARTICLES)
    run_trim_simulation(folder_base)
    mean_E, std_E, sem_E = extract_transmitted_energy(folder_base)

    folder_min = folder_base + "_min"
    print(f"running simulation for thickness {thickness} µm in folder {folder_min}")
    os.makedirs(folder_min, exist_ok=True)
    write_input_file(thickness - dx_err, folder_min, E + SIGMA_E, N_PARTICLES)
    run_trim_simulation(folder_min)
    mean_E_min, *_ = extract_transmitted_energy(folder_min)

    folder_max = folder_base + "_max"
    print(f"running simulation for thickness {thickness} µm in folder {folder_max}")
    os.makedirs(folder_max, exist_ok=True)
    write_input_file(thickness + dx_err, folder_max, E - SIGMA_E, N_PARTICLES)
    run_trim_simulation(folder_max)
    mean_E_max, *_ = extract_transmitted_energy(folder_max)

    sys_err = abs(mean_E_max - mean_E_min) / 2

    results.append({
        "thickness_um": thickness,
        "mean_exit_energy_MeV": mean_E,
        "sem_energy_MeV": sem_E,
        "sys_err_MeV": sys_err,
        "total_err_MeV": np.sqrt(sem_E**2 + sys_err**2)
    })

    # save in every iteration
    # append the results to a DataFrame and save it as a CSV file
    df_results = pd.DataFrame(results)
    df_results.to_csv(f"simulation_results_partial.csv", index=False)



df_results = pd.DataFrame(results)
time = pd.Timestamp.now().strftime("%Y-%m-%d_%H:%M:%S")
df_results.to_csv(f"simulation_results_{time}.csv", index=False)


fig = go.Figure()

fig.add_trace(go.Scatter(
    x=df_results['thickness_um'],
    y=df_results['mean_exit_energy_MeV'],
    error_y=dict(
        type='data',
        array=df_results['total_err_MeV'],
        visible=True,
        color='red',
        thickness=2,
        width=3
    ),
    mode='markers+lines',
    marker=dict(
        size=8,
        color='blue',
        symbol='circle'
    ),
    line=dict(
        color='blue',
        width=2
    ),
    name='Exit Energy',
    hovertemplate='<b>Thickness:</b> %{x} µm<br>' +
                  '<b>Exit Energy:</b> %{y:.4f} MeV<br>' +
                  '<b>Total Error:</b> ±%{error_y.array:.4f} MeV<br>' +
                  '<extra></extra>'
))

# Update layout
fig.update_layout(
    title=dict(
        text='Proton Exit Energy vs Aluminum Foil Thickness<br><sub>SRIM-2013 Monte Carlo Simulation</sub>',
        x=0.5,
        font=dict(size=16)
    ),
    xaxis=dict(
        title='Aluminum Foil Thickness (µm)',
        showgrid=True,
        gridwidth=1,
        gridcolor='lightgray',
        zeroline=True,
        zerolinewidth=2,
        zerolinecolor='gray'
    ),
    yaxis=dict(
        title='Exit Energy (MeV)',
        showgrid=True,
        gridwidth=1,
        gridcolor='lightgray',
        zeroline=True,
        zerolinewidth=2,
        zerolinecolor='gray'
    ),
    plot_bgcolor='white',
    showlegend=True,
    legend=dict(
        x=0.98,
        y=0.98,
        xanchor='right',
        yanchor='top',
        bgcolor='rgba(255,255,255,0.8)',
        bordercolor='gray',
        borderwidth=1
    ),
    margin=dict(l=70, r=50, t=80, b=60),
    width=800,
    height=600
)

# Add annotations with simulation parameters
fig.add_annotation(
    x=0.02,
    y=0.98,
    xref='paper',
    yref='paper',
    text=f'Initial Energy: {E:.2f} ± {SIGMA_E:.2f} MeV<br>' +
         f'Particles: {N_PARTICLES}<br>' +
         f'Thickness Error: ±{dx_err} µm',
    showarrow=False,
    font=dict(size=10),
    align='left',
    bgcolor='rgba(255,255,255,0.8)',
    bordercolor='gray',
    borderwidth=1
)

# Save as HTML
fig.write_html("energy_vs_thickness_plot.html")
print("\nPlot saved as 'energy_vs_thickness_plot.html'")

# Display summary statistics
print("\n=== Simulation Results Summary ===")
for _, row in df_results.iterrows():
    print(f"Thickness: {row['thickness_um']:3.0f} µm -> "
          f"Exit Energy: {row['mean_exit_energy_MeV']:.4f} ± {row['total_err_MeV']:.4f} MeV "
          f"(stat: ±{row['sem_energy_MeV']:.4f}, sys: ±{row['sys_err_MeV']:.4f})")

print(f"\nTotal simulations completed: {len(df_results)}")
print(f"Energy loss range: {df_results['mean_exit_energy_MeV'].min():.4f} - {df_results['mean_exit_energy_MeV'].max():.4f} MeV")


