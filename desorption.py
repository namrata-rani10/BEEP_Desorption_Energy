import numpy as np
import math
# Same constants as before
amu_to_kg = 1.66053906660e-27  # amu to kg
angstrom_to_m = 1e-10  # Ångström to meters

# Atomic masses dictionary
atomic_masses = {'H': 1.00784, 'C': 12.0107, 'O': 15.999, 'N': 14.0067, 'S': 32.06}

def get_atomic_mass(symbol):
    """Retrieve atomic mass or ask the user if not found."""
    if symbol in atomic_masses:
        return atomic_masses[symbol] * amu_to_kg
    else:
        mass_amu = float(input(f"Enter the atomic mass of {symbol} in amu: "))
        return mass_amu * amu_to_kg

def parse_coordinates(input_string):
    """Parse atomic symbols and their xyz coordinates."""
    symbols, coordinates = [], []
    for line in input_string.strip().splitlines():
        parts = line.split()
        symbols.append(parts[0])
        coordinates.append(list(map(float, parts[1:])))
    return symbols, np.array(coordinates)

def align_to_z_axis(symbols, coordinates, threshold=1e-8):
    """Align the molecule along the z-axis and zero out small values across all axes."""
    masses = np.array([get_atomic_mass(sym) for sym in symbols])
    total_mass = np.sum(masses)

    # Center the molecule at the origin (set center of mass to [0, 0, 0])
    center_of_mass = np.sum(masses[:, np.newaxis] * coordinates, axis=0) / total_mass
    shifted_coords = coordinates - center_of_mass

    # Use SVD to find the best alignment axis
    _, _, vh = np.linalg.svd(shifted_coords)
    rotation_matrix = vh.T

    # Rotate the coordinates to align the molecule along the z-axis
    aligned_coords = np.dot(shifted_coords, rotation_matrix)

    # Zero out small values based on the threshold for all axes
    aligned_coords[np.abs(aligned_coords) < threshold] = 0.0

    return aligned_coords


def get_moments_of_inertia(symbols, coordinates):
    """Calculate the moments of inertia after alignment."""
    masses = np.array([get_atomic_mass(sym) for sym in symbols])
    coords = coordinates * angstrom_to_m  # Convert to meters

    # Initialize the inertia tensor
    I = np.zeros((3, 3))
    for m, r in zip(masses, coords):
        I[0, 0] += m * (r[1]**2 + r[2]**2)
        I[1, 1] += m * (r[0]**2 + r[2]**2)
        I[2, 2] += m * (r[0]**2 + r[1]**2)
        I[0, 1] -= m * r[0] * r[1]
        I[0, 2] -= m * r[0] * r[2]
        I[1, 2] -= m * r[1] * r[2]

    # Fill symmetric elements
    I[1, 0], I[2, 0], I[2, 1] = I[0, 1], I[0, 2], I[1, 2]

    # Diagonalize the inertia tensor
    eigenvalues, _ = np.linalg.eigh(I)
    Ia, Ib, Ic = np.sort(eigenvalues)

    return Ia, Ib, Ic
def pre_exponential_factor(m, T, sigma, Ia, Ib, Ic):
    """Calculate the pre-exponential factor (v) for desorption."""
    kB = 1.380649e-23  # Boltzmann constant in J/K
    h = 6.62607015e-34  # Planck's constant in J·s
    pi = math.pi

    # Translational contribution
    translational_part = ((2 * pi * m * kB * T) / h**2)**(3 / 2)

    # Rotational contribution (considering Ia = 0 for linear molecules)
    if Ia == 0:
        rotational_part = (8 * pi**2 * kB * T / h**2) * (Ib / sigma)
    else:
        rotational_part = (pi**0.5 / sigma)*(8 * pi**2 * kB * T / h**2)**(3 / 2) * math.sqrt(Ia * Ib * Ic)

    # Final pre-exponential factor
    v = ((kB * T) / h) * translational_part  * rotational_part

    return v

def main():
    """Main function to get input and calculate the pre-exponential factor."""
    print("Enter the coordinates in the xyz format:")
    print("... (One atom per line)")
    print("When done, press Enter twice.\n")

    input_string = ""
    while True:
        line = input()
        if not line.strip():
            break
        input_string += line + "\n"

    symbols, coordinates = parse_coordinates(input_string)

    # Align the molecule along the z-axis
    aligned_coordinates = align_to_z_axis(symbols, coordinates)

    # Calculate moments of inertia
    Ia, Ib, Ic = get_moments_of_inertia(symbols, aligned_coordinates)
    print(f"Aligned coordinates:\n{aligned_coordinates}")
    print(f"Principal moments of inertia (kg·m²): Ia={Ia:.3e}, Ib={Ib:.3e}, Ic={Ic:.3e}")
   
    # Get additional input for the pre-exponential factor
    mass = float(input("Enter the mass of the molecule (in kg): "))
    temperature = float(input("Enter the temperature (in K): "))
    sigma = int(input("Enter the symmetry number of the molecule: "))

    # Calculate the pre-exponential factor
    v = pre_exponential_factor(mass, temperature, sigma, Ia, Ib, Ic)
    print(f"Pre-exponential factor (v): {v:.3e} s⁻¹")

if __name__ == "__main__":
    main()
