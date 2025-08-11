import numpy as np # type: ignore
import pandas as pd # type: ignore
import plotly.graph_objects as go # type: ignore
from tqdm import tqdm # type: ignore
from uncertainties import ufloat # type: ignore
import os
import shutil
import subprocess
from typing import Literal
from pathlib import Path
from itertools import product


def extract_transmitted_energy(folder):
    filepath = os.path.join(folder, "TRANSMIT.txt")
    if not os.path.exists(filepath):
        return np.nan, np.nan, np.nan, np.array([])
    energies = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.split()
            try:
                if line.startswith("T "):
                    energy_ev = float(parts[3])  # energy is on 4th column in units of eV

                elif line.startswith("T"):
                    energy_ev = float(parts[2])  # energy is on 3rd column in units of eV
                else:
                    continue
            except (ValueError, IndexError):
                continue

            energy_mev = energy_ev / 1e6  # convert to MeV
            energies.append(energy_mev)
    if energies:
        energies = np.array(energies)
        mean = np.mean(energies)
        stde = np.std(energies, ddof=1)
        sem = stde / np.sqrt(len(energies))
        return mean, stde, sem, energies
    return np.nan, np.nan, np.nan, np.array([])

file_directory = Path(__file__).parent

mean_E, std_E, sem_E, energies = extract_transmitted_energy(file_directory)

print(f"Mean Energy: {mean_E:.2f} MeV")
print(f"Standard Deviation: {std_E:.2f} MeV")
print(f"Standard Error of the Mean: {sem_E:.2f} MeV")

