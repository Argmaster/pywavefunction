from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Atom:
    """Dataclass representing an atom."""

    symbol: str
    """The atomic symbol of the atom."""
    atomic_number: int
    """The atomic number of the atom."""
    mass: float
    """The atomic mass of the atom."""

    def __str__(self) -> str:
        return self.symbol


H = Atom("H", 1, 1.00784)
"""Hydrogen atom."""

He = Atom("He", 2, 4.002602)
"""Helium atom."""

Li = Atom("Li", 3, 6.938)
"""Lithium atom."""

Be = Atom("Be", 4, 9.0121831)
"""Beryllium atom."""

B = Atom("B", 5, 10.81)
"""Boron atom."""

C = Atom("C", 6, 12.011)
"""Carbon atom."""

N = Atom("N", 7, 14.007)
"""Nitrogen atom."""

O = Atom("O", 8, 15.999)
"""Oxygen atom."""

F = Atom("F", 9, 18.998403163)
"""Fluorine atom."""

Ne = Atom("Ne", 10, 20.1797)
"""Neon atom."""

Na = Atom("Na", 11, 22.98976928)
"""Sodium atom."""

Mg = Atom("Mg", 12, 24.305)
"""Magnesium atom."""

Al = Atom("Al", 13, 26.9815385)
"""Aluminium atom."""

Si = Atom("Si", 14, 28.085)
"""Silicon atom."""

P = Atom("P", 15, 30.973761998)
"""Phosphorus atom."""

S = Atom("S", 16, 32.06)

Cl = Atom("Cl", 17, 35.45)

Ar = Atom("Ar", 18, 39.948)
"""Argon atom."""

K = Atom("K", 19, 39.0983)
"""Potassium atom."""

Ca = Atom("Ca", 20, 40.078)
"""Calcium atom."""

Sc = Atom("Sc", 21, 44.955908)
"""Scandium atom."""

Ti = Atom("Ti", 22, 47.867)
"""Titanium atom."""

V = Atom("V", 23, 50.9415)
"""Vanadium atom."""

Cr = Atom("Cr", 24, 51.9961)
"""Chromium atom."""

Mn = Atom("Mn", 25, 54.938044)
"""Manganese atom."""

Fe = Atom("Fe", 26, 55.845)
"""Iron atom."""

Co = Atom("Co", 27, 58.933194)
"""Cobalt atom."""

Ni = Atom("Ni", 28, 58.6934)
"""Nickel atom."""

Cu = Atom("Cu", 29, 63.546)
"""Copper atom."""

Zn = Atom("Zn", 30, 65.38)
"""Zinc atom."""

Ga = Atom("Ga", 31, 69.723)
"""Gallium atom."""

Ge = Atom("Ge", 32, 72.63)
"""Germanium atom."""

As = Atom("As", 33, 74.921595)
"""Arsenic atom."""

Se = Atom("Se", 34, 78.971)
"""Selenium atom."""

Br = Atom("Br", 35, 79.904)
"""Bromine atom."""

Kr = Atom("Kr", 36, 83.798)
"""Krypton atom."""

Rb = Atom("Rb", 37, 85.4678)
"""Rubidium atom."""

Sr = Atom("Sr", 38, 87.62)
"""Strontium atom."""

Y = Atom("Y", 39, 88.90584)
"""Yttrium atom."""

Zr = Atom("Zr", 40, 91.224)
"""Zirconium atom."""

Nb = Atom("Nb", 41, 92.90637)
"""Niobium atom."""

Mo = Atom("Mo", 42, 95.95)
"""Molybdenum atom."""

Tc = Atom("Tc", 43, 98)
"""Technetium atom."""

Ru = Atom("Ru", 44, 101.07)
"""Ruthenium atom."""

Rh = Atom("Rh", 45, 102.90550)
"""Rhodium atom."""

Pd = Atom("Pd", 46, 106.42)
"""Palladium atom."""

Ag = Atom("Ag", 47, 107.8682)
"""Silver atom."""

Cd = Atom("Cd", 48, 112.414)
"""Cadmium atom."""

In = Atom("In", 49, 114.818)
"""Indium atom."""

Sn = Atom("Sn", 50, 118.710)
"""Tin atom."""

Sb = Atom("Sb", 51, 121.760)
"""Antimony atom."""

Te = Atom("Te", 52, 127.60)
"""Tellurium atom."""

I = Atom("I", 53, 126.90447)
"""Iodine atom."""

Xe = Atom("Xe", 54, 131.293)
"""Xenon atom."""

Cs = Atom("Cs", 55, 132.90545196)
"""Caesium atom."""

Ba = Atom("Ba", 56, 137.327)
"""Barium atom."""

La = Atom("La", 57, 138.90547)
"""Lanthanum atom."""

Ce = Atom("Ce", 58, 140.116)
"""Cerium atom."""

Pr = Atom("Pr", 59, 140.90766)
"""Praseodymium atom."""

Nd = Atom("Nd", 60, 144.242)
"""Neodymium atom."""

Pm = Atom("Pm", 61, 145)
"""Promethium atom."""

Sm = Atom("Sm", 62, 150.36)
"""Samarium atom."""

Eu = Atom("Eu", 63, 151.964)
"""Europium atom."""

Gd = Atom("Gd", 64, 157.25)
"""Gadolinium atom."""

Tb = Atom("Tb", 65, 158.92535)
"""Terbium atom."""

Dy = Atom("Dy", 66, 162.500)
"""Dysprosium atom."""

Ho = Atom("Ho", 67, 164.93033)
"""Holmium atom."""

Er = Atom("Er", 68, 167.259)
"""Erbium atom."""

Tm = Atom("Tm", 69, 168.93422)
"""Thulium atom."""

Yb = Atom("Yb", 70, 173.054)
"""Ytterbium atom."""

Lu = Atom("Lu", 71, 174.9668)
"""Lutetium atom."""

Hf = Atom("Hf", 72, 178.49)
"""Hafnium atom."""

Ta = Atom("Ta", 73, 180.94788)
"""Tantalum atom."""

W = Atom("W", 74, 183.84)
"""Tungsten atom."""

Re = Atom("Re", 75, 186.207)
"""Rhenium atom."""

Os = Atom("Os", 76, 190.23)
"""Osmium atom."""

Ir = Atom("Ir", 77, 192.217)
"""Iridium atom."""

Pt = Atom("Pt", 78, 195.084)
"""Platinum atom."""

Au = Atom("Au", 79, 196.966569)
"""Gold atom."""

Hg = Atom("Hg", 80, 200.592)
"""Mercury atom."""

Tl = Atom("Tl", 81, 204.38)
"""Thallium atom."""

Pb = Atom("Pb", 82, 207.2)
"""Lead atom."""

Bi = Atom("Bi", 83, 208.98040)
"""Bismuth atom."""

Po = Atom("Po", 84, 209)
"""Polonium atom."""

At = Atom("At", 85, 210)
"""Astatine atom."""

Rn = Atom("Rn", 86, 222)
"""Radon atom."""

Fr = Atom("Fr", 87, 223)
"""Francium atom."""

Ra = Atom("Ra", 88, 226)
"""Radium atom."""

Ac = Atom("Ac", 89, 227)
"""Actinium atom."""

Th = Atom("Th", 90, 232.0377)
"""Thorium atom."""

Pa = Atom("Pa", 91, 231.03588)
"""Protactinium atom."""

U = Atom("U", 92, 238.02891)
"""Uranium atom."""

Np = Atom("Np", 93, 237)
"""Neptunium atom."""

Pu = Atom("Pu", 94, 244)
"""Plutonium atom."""

Am = Atom("Am", 95, 243)
"""Americium atom."""

Cm = Atom("Cm", 96, 247)
"""Curium atom."""

Bk = Atom("Bk", 97, 247)
"""Berkelium atom."""

Cf = Atom("Cf", 98, 251)
"""Californium atom."""

Es = Atom("Es", 99, 252)
"""Einsteinium atom."""

Fm = Atom("Fm", 100, 257)
"""Fermium atom."""

Md = Atom("Md", 101, 258)
"""Mendelevium atom."""

No = Atom("No", 102, 259)
"""Nobelium atom."""

Lr = Atom("Lr", 103, 266)
"""Lawrencium atom."""

Rf = Atom("Rf", 104, 267)
"""Rutherfordium atom."""

Db = Atom("Db", 105, 270)
"""Dubnium atom."""

Sg = Atom("Sg", 106, 271)
"""Seaborgium atom."""

Bh = Atom("Bh", 107, 270)
"""Bohrium atom."""

Hs = Atom("Hs", 108, 277)
"""Hassium atom."""

Mt = Atom("Mt", 109, 276)
"""Meitnerium atom."""

Ds = Atom("Ds", 110, 281)
"""Darmstadtium atom."""

Rg = Atom("Rg", 111, 280)
"""Roentgenium atom."""

Cn = Atom("Cn", 112, 285)
"""Copernicium atom."""

Nh = Atom("Nh", 113, 284)
"""Nihonium atom."""

Fl = Atom("Fl", 114, 289)
"""Flerovium atom."""

Mc = Atom("Mc", 115, 288)
"""Moscovium atom."""

Lv = Atom("Lv", 116, 293)
"""Livermorium atom."""

Ts = Atom("Ts", 117, 294)
"""Tennessine atom."""

Og = Atom("Og", 118, 294)
"""Oganesson atom."""
