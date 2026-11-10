"""The OpenMM force of PolarMACE models (symmetrix.openmm_ml) against symmetrix.experimental.MACEPolar (ASE), which
test_mace_polar.py compares with PyTorch, for the model in $SYMMETRIX_POLAR_MODEL. Skipped without it."""

import numpy as np
import pytest

openmm = pytest.importorskip("openmm")
pytest.importorskip("openmmml")
from openmm import app, unit  # noqa: E402
from openmmml import MLPotential  # noqa: E402

from symmetrix.experimental import MACEPolar  # noqa: E402
from symmetrix.openmm_ml import EV_TO_KJ_PER_MOL, create_mace_polar_force  # noqa: E402
from test_mace_polar import MODEL, gpu_available, polar, structures  # noqa: E402, F401

devices = [
    pytest.param("cpu", "double", 1e-12, 1e-6, id="cpu-double"),
    pytest.param(
        "cuda",
        "single",
        1e-8,
        1e-2,
        id="cuda-single",
        marks=pytest.mark.skipif(not gpu_available(), reason="no CUDA build or GPU"),
    ),
]


def topology(a):
    top = app.Topology()
    residue = top.addResidue("MOL", top.addChain())
    for z in a.numbers:
        element = app.Element.getByAtomicNumber(int(z))
        top.addAtom(element.symbol, element, residue)
    top.setPeriodicBoxVectors(
        [openmm.Vec3(*v) for v in a.cell.array * 0.1] * unit.nanometer
    )
    return top


def evaluate(system, a, platform="Reference"):
    context = openmm.Context(
        system,
        openmm.VerletIntegrator(0.0005),
        openmm.Platform.getPlatformByName(platform),
    )
    context.setPeriodicBoxVectors(*[openmm.Vec3(*v) for v in a.cell.array * 0.1])
    context.setPositions(a.positions * 0.1)
    state = context.getState(getEnergy=True, getForces=True)
    return (
        state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole),
        state.getForces(asNumpy=True).value_in_unit(
            unit.kilojoule_per_mole / unit.nanometer
        ),
    )


def ase_reference(model_json, a, device, precision):
    b = a.copy()
    b.calc = MACEPolar(
        model_json,
        device=device,
        dtype={"double": "float64", "single": "float32"}[precision],
    )
    return (
        b.get_potential_energy() * EV_TO_KJ_PER_MOL,
        b.get_forces() * 10 * EV_TO_KJ_PER_MOL,
    )


@pytest.mark.parametrize("name", ["water", "triclinic", "hydroxide", "radical"])
@pytest.mark.parametrize("device,precision,e_rtol,f_tol", devices)
def test_matches_ase(polar, name, device, precision, e_rtol, f_tol):
    a = structures()[name]
    system = MLPotential("symmetrix-mace-polar", modelPath=polar).createSystem(
        topology(a),
        charge=a.info.get("charge", 0),
        multiplicity=a.info.get("spin", 1),
        device=device,
        precision=precision,
    )
    energy, forces = evaluate(system, a)
    e_ref, f_ref = ase_reference(polar, a, device, precision)
    print(
        f"{name} {device}: dE {abs(energy - e_ref):.2e} kJ/mol of {e_ref:.6e}, max dF {np.abs(forces - f_ref).max():.2e}"
    )
    assert abs(energy - e_ref) < e_rtol * abs(e_ref)
    assert np.abs(forces - f_ref).max() < f_tol


def test_particles_subset(polar):
    """the force on a subset of the System's particles, with the others outside the model"""
    a = structures()["hydroxide"]
    system = openmm.System()
    system.setDefaultPeriodicBoxVectors(*[openmm.Vec3(*v) for v in a.cell.array * 0.1])
    for _ in range(len(a) + 2):
        system.addParticle(1.0)
    particles = list(
        range(1, len(a) + 1)
    )  # particles 0 and len(a)+1 are not in the model
    system.addForce(
        create_mace_polar_force(
            polar, a.numbers, device="cpu", charge=-1, particles=particles
        )
    )
    extra = np.vstack([[0.1, 0.2, 0.3], a.positions, [1.0, 2.0, 3.0]])
    context = openmm.Context(
        system,
        openmm.VerletIntegrator(0.0005),
        openmm.Platform.getPlatformByName("Reference"),
    )
    context.setPositions(extra * 0.1)
    state = context.getState(getEnergy=True, getForces=True)
    forces = state.getForces(asNumpy=True).value_in_unit(
        unit.kilojoule_per_mole / unit.nanometer
    )
    e_ref, f_ref = ase_reference(polar, a, "cpu", "double")
    assert abs(
        state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole) - e_ref
    ) < 1e-12 * abs(e_ref)
    assert np.abs(forces[1:-1] - f_ref).max() < 1e-6
    assert np.all(forces[[0, -1]] == 0)


def test_xml_serialization(polar):
    a = structures()["triclinic"]
    system = MLPotential("symmetrix-mace-polar", modelPath=polar).createSystem(
        topology(a), device="cpu"
    )
    copy = openmm.XmlSerializer.deserialize(openmm.XmlSerializer.serialize(system))
    e, f = evaluate(system, a)
    e2, f2 = evaluate(copy, a)
    assert abs(e - e2) < 1e-12 * abs(e)
    assert np.abs(f - f2).max() < 1e-6


def test_extracts_pytorch_model(polar, tmp_path, monkeypatch):
    """modelPath may be the PyTorch model; it is extracted for the Topology's elements into openmm-ml's cache"""
    monkeypatch.setenv("HOME", str(tmp_path))
    a = structures()["triclinic"]
    system = MLPotential("symmetrix-mace-polar", modelPath=MODEL).createSystem(
        topology(a), device="cpu"
    )
    assert len(list((tmp_path / ".cache" / "openmm-ml").glob("symmetrix-*.json"))) == 1
    e, f = evaluate(system, a)
    e_ref, f_ref = ase_reference(polar, a, "cpu", "double")
    assert abs(e - e_ref) < 1e-12 * abs(e_ref)
    assert np.abs(f - f_ref).max() < 1e-6


def test_nve_energy_conservation(polar):
    a = structures()["triclinic"]
    system = MLPotential("symmetrix-mace-polar", modelPath=polar).createSystem(
        topology(a), device="cpu"
    )
    integrator = openmm.VerletIntegrator(0.25 * unit.femtoseconds)
    context = openmm.Context(
        system, integrator, openmm.Platform.getPlatformByName("Reference")
    )
    context.setPositions(a.positions * 0.1)
    context.setVelocitiesToTemperature(300 * unit.kelvin, 1)
    totals, kinetic = [], []
    for _ in range(40):
        integrator.step(5)
        s = context.getState(getEnergy=True)
        k = s.getKineticEnergy().value_in_unit(unit.kilojoule_per_mole)
        totals.append(s.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole) + k)
        kinetic.append(k)
    print(
        f"total energy range {np.ptp(totals):.3e} kJ/mol, mean kinetic {np.mean(kinetic):.3e} kJ/mol"
    )
    assert np.ptp(totals) < 1e-2 * np.mean(kinetic)


def test_rejects_non_periodic_and_interaction_energy(polar):
    a = structures()["water"]
    top = topology(a)
    potential = MLPotential("symmetrix-mace-polar", modelPath=polar)
    with pytest.raises(ValueError, match="returnEnergyType"):
        potential.createSystem(top, device="cpu", returnEnergyType="interaction_energy")
    top.setPeriodicBoxVectors(None)
    with pytest.raises(ValueError, match="periodic"):
        potential.createSystem(top, device="cpu")
