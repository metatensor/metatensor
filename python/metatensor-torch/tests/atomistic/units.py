from metatensor.torch.atomistic.units import KNOWN_QUANTITIES


def test_conversion_length():

    length = KNOWN_QUANTITIES["length"]
    length_angstrom = 1.0
    length_nm = length.conversion("angstrom", "nm") * length_angstrom
    assert length_nm == 0.1


def test_conversion_energy():

    energy = KNOWN_QUANTITIES["energy"]
    energy_ev = 1.0
    energy_mev = energy.conversion("ev", "mev") * energy_ev
    assert energy_mev == 1000.0
