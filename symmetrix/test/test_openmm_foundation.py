"""The OpenMM forces of the models of pair_style symmetrix/mace/foundation and symmetrix/nequip/foundation
(symmetrix.openmm_ml) against symmetrix.experimental.MACEFoundation and NequIPFoundation (ASE), which
test_mace_foundation.py and test_nequip_foundation.py compare with PyTorch.

The MACE model is $SYMMETRIX_MACE_MODEL, or MACE-OMAT-0 small from mace-torch's cache (downloaded if absent). The
NequIP model is extracted from NequIP-OAM-S when nequip is installed. Each is skipped without its dependencies."""

import json

import numpy as np
import pytest

openmm = pytest.importorskip("openmm")
pytest.importorskip("openmmml")
from openmm import app, unit  # noqa: E402
from openmmml import MLPotential  # noqa: E402

from symmetrix.experimental import MACEFoundation, NequIPFoundation  # noqa: E402
from symmetrix.openmm_ml import (  # noqa: E402
    EV_TO_KJ_PER_MOL,
    create_mace_foundation_force,
)
from test_mace_foundation import structures  # noqa: E402

CALCULATORS = {"mace": MACEFoundation, "nequip": NequIPFoundation}
POTENTIALS = {
    "mace": "symmetrix-mace-foundation",
    "nequip": "symmetrix-nequip-foundation",
}


def _mace_model():
    import os

    path = os.environ.get("SYMMETRIX_MACE_MODEL")
    if path:
        return path
    try:
        from mace.calculators.foundations_models import download_mace_mp_checkpoint
    except ImportError:
        pytest.skip("mace-torch is not available")
    return str(download_mace_mp_checkpoint("small-omat-0"))


@pytest.fixture(scope="module")
def mace_model(tmp_path_factory):
    try:
        from symmetrix.extract_mace_data import extract_mace_data
    except ImportError:
        pytest.skip("mace-torch is not available")
    model = _mace_model()
    model_json = tmp_path_factory.mktemp("mace") / "mace-1-8.json"
    model_json.write_text(json.dumps(extract_mace_data(model, [1, 8])))
    return model, str(model_json)


@pytest.fixture(scope="module")
def nequip_model(tmp_path_factory):
    pytest.importorskip("nequip")
    from symmetrix.extract_nequip_data import extract_nequip_data

    model = "nequip.net:mir-group/NequIP-OAM-S:0.1"
    model_json = tmp_path_factory.mktemp("nequip") / "nequip-1-8.json"
    model_json.write_text(json.dumps(extract_nequip_data(model, [1, 8])))
    return model, str(model_json)


@pytest.fixture(params=["mace", "nequip"])
def model(request):
    """(kind, PyTorch model, Symmetrix JSON)"""
    return (request.param, *request.getfixturevalue(f"{request.param}_model"))


def calculator(kind, model_json, device, dtype):
    try:
        return CALCULATORS[kind](model_json, device=device, dtype=dtype)
    except RuntimeError as e:
        no_gpu = ["built without CUDA", "no CUDA-capable device", "driver version is"]
        if device == "cuda" and any(m in str(e) for m in no_gpu):
            pytest.skip(str(e))
        raise


def topology(a):
    top = app.Topology()
    residue = top.addResidue("MOL", top.addChain())
    for z in a.numbers:
        element = app.Element.getByAtomicNumber(int(z))
        top.addAtom(element.symbol, element, residue)
    if a.pbc.all():
        top.setPeriodicBoxVectors(
            [openmm.Vec3(*v) for v in a.cell.array * 0.1] * unit.nanometer
        )
    return top


def context(system, a):
    c = openmm.Context(
        system,
        openmm.VerletIntegrator(0.25 * unit.femtoseconds),
        openmm.Platform.getPlatformByName("Reference"),
    )
    if a.pbc.all():
        c.setPeriodicBoxVectors(*[openmm.Vec3(*v) for v in a.cell.array * 0.1])
    c.setPositions(a.positions * 0.1)
    return c


def energy_forces(c):
    state = c.getState(getEnergy=True, getForces=True, getPositions=True)
    return (
        state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole),
        state.getForces(asNumpy=True).value_in_unit(
            unit.kilojoule_per_mole / unit.nanometer
        ),
        state.getPositions(asNumpy=True).value_in_unit(unit.angstrom),
    )


def ase_reference(kind, model_json, a, device, dtype):
    b = a.copy()
    b.calc = calculator(kind, model_json, device, dtype)
    return (
        b.get_potential_energy() * EV_TO_KJ_PER_MOL,
        b.get_forces() * 10 * EV_TO_KJ_PER_MOL,
    )


devices = [
    pytest.param("cpu", "double", "float64", 1e-12, 1e-6, id="cpu-double"),
    pytest.param("cpu", "single", "float32", 1e-6, 1e-2, id="cpu-single"),
    pytest.param("cuda", "double", "float64", 1e-12, 1e-6, id="cuda-double"),
    pytest.param("cuda", "single", "float32", 1e-6, 1e-2, id="cuda-single"),
    pytest.param("cuda", "mixed", "float32-mixed", 1e-6, 1e-2, id="cuda-mixed"),
]


@pytest.mark.parametrize("name", ["bulk", "cluster"])
@pytest.mark.parametrize("device,precision,dtype,e_rtol,f_tol", devices)
def test_matches_ase(model, name, device, precision, dtype, e_rtol, f_tol):
    kind, _, model_json = model
    a = structures()[name]
    e_ref, f_ref = ase_reference(kind, model_json, a, device, dtype)
    system = MLPotential(POTENTIALS[kind], modelPath=model_json).createSystem(
        topology(a), device=device, precision=precision
    )
    energy, forces, _ = energy_forces(context(system, a))
    print(
        f"{kind} {name} {device}-{precision}: dE {abs(energy - e_ref):.2e} kJ/mol of {e_ref:.6e}, "
        f"max dF {np.abs(forces - f_ref).max():.2e}"
    )
    assert abs(energy - e_ref) < e_rtol * abs(e_ref)
    assert np.abs(forces - f_ref).max() < f_tol


def test_neighbor_list_after_steps(model):
    """after enough MD steps for the skin to be crossed, the force still matches a fresh evaluation"""
    kind, _, model_json = model
    a = structures()["bulk"]
    system = MLPotential(POTENTIALS[kind], modelPath=model_json).createSystem(
        topology(a), device="cpu", skin=0.2
    )
    c = context(system, a)
    c.setVelocitiesToTemperature(2000 * unit.kelvin, 1)
    c.getIntegrator().step(200)
    energy, forces, x = energy_forces(c)
    b = a.copy()
    b.positions = x
    e_ref, f_ref = ase_reference(kind, model_json, b, "cpu", "float64")
    assert np.abs(x - a.positions).max() > 0.2
    assert abs(energy - e_ref) < 1e-12 * abs(e_ref)
    assert np.abs(forces - f_ref).max() < 1e-6


def test_particles_subset(mace_model):
    """the force on a subset of the System's particles, with the others outside the model"""
    _, model_json = mace_model
    a = structures()["cluster"]
    system = openmm.System()
    for _ in range(len(a) + 2):
        system.addParticle(1.0)
    particles = list(range(1, len(a) + 1))
    system.addForce(
        create_mace_foundation_force(
            model_json, a.numbers, device="cpu", periodic=False, particles=particles
        )
    )
    c = openmm.Context(
        system,
        openmm.VerletIntegrator(0.0005),
        openmm.Platform.getPlatformByName("Reference"),
    )
    c.setPositions(np.vstack([[30.0, 0, 0], a.positions, [-30.0, 0, 0]]) * 0.1)
    energy, forces, _ = energy_forces(c)
    e_ref, f_ref = ase_reference("mace", model_json, a, "cpu", "float64")
    assert abs(energy - e_ref) < 1e-12 * abs(e_ref)
    assert np.abs(forces[1:-1] - f_ref).max() < 1e-6
    assert np.all(forces[[0, -1]] == 0)


def test_xml_serialization(mace_model):
    _, model_json = mace_model
    a = structures()["bulk"]
    system = MLPotential(POTENTIALS["mace"], modelPath=model_json).createSystem(
        topology(a), device="cpu"
    )
    copy = openmm.XmlSerializer.deserialize(openmm.XmlSerializer.serialize(system))
    e, f, _ = energy_forces(context(system, a))
    e2, f2, _ = energy_forces(context(copy, a))
    assert abs(e - e2) < 1e-12 * abs(e)
    assert np.abs(f - f2).max() < 1e-6


def test_extracts_pytorch_model(mace_model, tmp_path, monkeypatch):
    """modelPath may be the PyTorch model; it is extracted for the Topology's elements into openmm-ml's cache"""
    model, model_json = mace_model
    monkeypatch.setenv("HOME", str(tmp_path))
    a = structures()["cluster"]
    system = MLPotential(POTENTIALS["mace"], modelPath=model).createSystem(
        topology(a), device="cpu"
    )
    assert len(list((tmp_path / ".cache" / "openmm-ml").glob("symmetrix-*.json"))) == 1
    e, f, _ = energy_forces(context(system, a))
    e_ref, f_ref = ase_reference("mace", model_json, a, "cpu", "float64")
    assert abs(e - e_ref) < 1e-12 * abs(e_ref)
    assert np.abs(f - f_ref).max() < 1e-6


def test_nve_energy_conservation(mace_model):
    _, model_json = mace_model
    a = structures()["cluster"]
    system = MLPotential(POTENTIALS["mace"], modelPath=model_json).createSystem(
        topology(a), device="cpu"
    )
    c = context(system, a)
    c.setVelocitiesToTemperature(300 * unit.kelvin, 1)
    totals, kinetic = [], []
    for _ in range(40):
        c.getIntegrator().step(5)
        s = c.getState(getEnergy=True)
        k = s.getKineticEnergy().value_in_unit(unit.kilojoule_per_mole)
        totals.append(s.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole) + k)
        kinetic.append(k)
    print(
        f"total energy range {np.ptp(totals):.3e} kJ/mol, mean kinetic {np.mean(kinetic):.3e} kJ/mol"
    )
    assert np.ptp(totals) < 1e-2 * np.mean(kinetic)


def test_rejects_interaction_energy(mace_model):
    _, model_json = mace_model
    potential = MLPotential(POTENTIALS["mace"], modelPath=model_json)
    with pytest.raises(ValueError, match="returnEnergyType"):
        potential.createSystem(
            topology(structures()["bulk"]),
            device="cpu",
            returnEnergyType="interaction_energy",
        )
