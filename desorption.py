import math
import numpy as np

# Atomic masses in atomic mass units (amu) for common elements
atomic_masses = {
    'H': 1.00784, 'C': 12.0107, 'O': 15.999, 'N': 14.0067, 'S': 32.06,
}

amu_to_kg = 1.66053906660e-27  # Conversion factor from amu to kg
angstrom_to_m = 1e-10  # Conversion factor from Å to m

def get_atomic_mass(symbol):
    """Retrieve atomic mass from the dictionary or asking the user."""
    if symbol in atomic_masses:
        return atomic_masses[symbol] * amu_to_kg
    else:
        print(f"Atomic mass for '{symbol}' not found.")
        mass_amu = float(input(f"Please enter the atomic mass of {symbol} in amu: "))
        return mass_amu * amu_to_kg

def parse_coordinates(input_string):
    """Parse atomic symbols and coordinates from a multiline string."""
    symbols = []
    coordinates = []

    # Split input into lines and extract symbol + coordinates from each line
    for line in input_string.strip().splitlines():
        parts = line.split()
        symbol = parts[0]
        x, y, z = map(float, parts[1:])
        symbols.append(symbol)
        coordinates.append([x, y, z])

    return symbols, coordinates

def get_moments_of_inertia(symbols, coordinates):
    """Calculate the principal moments of inertia from symbols and coordinates."""
    masses = np.array([get_atomic_mass(sym) for sym in symbols])
    coords = np.array(coordinates) * angstrom_to_m

    # Calculate the center of mass
    total_mass = np.sum(masses)
    center_of_mass = np.sum(masses[:, np.newaxis] * coords, axis=0) / total_mass

    # Shift coordinates to the center of mass
    shifted_coords = coords - center_of_mass

    # Calculate the moment of inertia tensor
    I = np.zeros((3, 3))
    for m, r in zip(masses, shifted_coords):
        I[0, 0] += m * (r[1]**2 + r[2]**2)
        I[1, 1] += m * (r[0]**2 + r[2]**2)
        I[2, 2] += m * (r[0]**2 + r[1]**2)
        I[0, 1] -= m * r[0] * r[1]
        I[0, 2] -= m * r[0] * r[2]
        I[1, 2] -= m * r[1] * r[2]

    I[1, 0] = I[0, 1]
    I[2, 0] = I[0, 2]
    I[2, 1] = I[1, 2]

    # Diagonalize the moment of inertia tensor to get principal moments
    eigenvalues, _ = np.linalg.eigh(I)
    Ia, Ib, Ic = np.sort(eigenvalues)  # Sort moments of inertia

    # Adjust for linear molecules
    if len(symbols) <= 3:  # Assuming linear for diatomic and triatomic
        Ia = 0  # Set the moment of inertia along the molecular axis to zero

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
        rotational_part = (8 * pi**2 * kB * T / h**2)**(3 / 2) * math.sqrt(Ib * Ic)
    else:
        rotational_part = (8 * pi**2 * kB * T / h**2)**(3 / 2) * math.sqrt(Ia * Ib * Ic)

    # Final pre-exponential factor
    v = ((kB * T) / h) * translational_part * (pi**0.5 / sigma) * rotational_part

    return v

def main():
    """Main function to get input and calculate the pre-exponential factor."""
    print("Enter the coordinates in the following format:")

    # Read multi-line cartesian input from user
    input_string = ""
    while True:
        line = input()
        if line.strip() == "":
            break
        input_string += line + "\n"

    # Parse symbols and coordinates
    symbols, coordinates = parse_coordinates(input_string)

    # Calculate moments of inertia
    Ia, Ib, Ic = get_moments_of_inertia(symbols, coordinates)
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
