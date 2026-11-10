"""PolarMACE models in OpenMM, through an openmm.PythonForce.

With openmm-ml installed, the potential is registered as "symmetrix-mace-polar":

    from openmmml import MLPotential
    potential = MLPotential("symmetrix-mace-polar", modelPath="polar-m-H-O.json")
    system = potential.createSystem(topology, charge=0, multiplicity=1)

Without openmm-ml, create_mace_polar_force() returns the Force to add to a System.
"""

import hashlib
import os

import numpy as np
import openmm
from openmm import unit

from .experimental import MACEPolarEvaluator

# e N_A / 1000, exact in the SI since 2019
EV_TO_KJ_PER_MOL = 1.602176634e-19 * 6.02214076e23 / 1000


class MACEPolarComputation:
    """The computation function of the PythonForce: energy (kJ/mol) and forces (kJ/mol/nm) of the particles of an
    OpenMM State. It pickles as its arguments, so XmlSerializer can save the force; the evaluator is created again
    on first use."""

    def __init__(
        self,
        model_file,
        atomic_numbers,
        device="cuda",
        dtype=None,
        charge=0.0,
        multiplicity=1.0,
    ):
        self.args = dict(
            model_file=str(model_file),
            atomic_numbers=[int(z) for z in atomic_numbers],
            device=device,
            dtype=dtype,
            charge=float(charge),
            multiplicity=float(multiplicity),
        )
        self.polar = None

    def setup(self):
        a = self.args
        self.polar = MACEPolarEvaluator(a["model_file"], a["device"], a["dtype"])
        self.node_types = self.polar.node_types(a["atomic_numbers"])

    def __getstate__(self):
        return self.args

    def __setstate__(self, args):
        self.args = args
        self.polar = None

    def __call__(self, state):
        if self.polar is None:
            self.setup()
        x = state.getPositions(asNumpy=True).value_in_unit(unit.angstrom)
        cell = state.getPeriodicBoxVectors(asNumpy=True).value_in_unit(unit.angstrom)
        energies, forces, _ = self.polar.compute(
            x, self.node_types, cell, self.args["charge"], self.args["multiplicity"]
        )
        return float(np.sum(energies)) * EV_TO_KJ_PER_MOL, forces * (
            10 * EV_TO_KJ_PER_MOL
        )


def create_mace_polar_force(
    model_file,
    atomic_numbers,
    device="cuda",
    dtype=None,
    charge=0.0,
    multiplicity=1.0,
    particles=None,
):
    """A periodic openmm.PythonForce evaluating a PolarMACE model (Symmetrix JSON) with MACEPolarEvaluator.

    atomic_numbers are those of the particles it applies to: all of the System's, or those in `particles`. The energy
    is the model's total energy, isolated-atom energies included. The System must be periodic: the electrostatics
    are k-space sums over the model's k-vectors."""
    compute = MACEPolarComputation(
        model_file, atomic_numbers, device, dtype, charge, multiplicity
    )
    compute.setup()  # fails here, not in the first step, on a bad model, device or element
    force = openmm.PythonForce(compute)
    force.setUsesPeriodicBoundaryConditions(True)
    if particles is not None:
        force.setParticles(list(particles))
    force.setName("SymmetrixMACEPolar")
    return force


def _cuda_built():
    try:
        from . import _mace_polar  # noqa: F401
    except ImportError:
        return False
    return True


try:
    from openmmml.mlpotential import MLPotentialImpl
except ImportError:  # openmm-ml is optional
    MLPotentialImpl = object


class SymmetrixMACEPolarImpl(MLPotentialImpl):
    """PolarMACE models evaluated by Symmetrix, for openmm-ml, where symmetrix_openmmml registers it.

    >>> potential = MLPotential("symmetrix-mace-polar", modelPath="polar-m-H-O.json")
    >>> system = potential.createSystem(topology, charge=-1, multiplicity=1, device="cuda", precision="single")

    modelPath is a Symmetrix JSON, or a PyTorch PolarMACE model, which is then extracted for the elements of the
    Topology (with the extraction's dependencies) into openmm-ml's cache directory.

    Arguments of createSystem() and createMixedSystem():

    - charge, multiplicity: total charge (0) and spin multiplicity (1) of the atoms the potential applies to.
    - device: "cuda" (the default when Symmetrix was built with CUDA) or "cpu". The GPU is the current CUDA device.
    - precision: "single" (GPU), "double" (CPU) or "mixed" (GPU, fp16 tensor-core GEMMs to float32 accuracy); or
      dtype as for MACEPolarEvaluator.
    - returnEnergyType: only "energy", the total energy with the isolated-atom energies. Symmetrix model files store
      those energies together with the model's energy shift, so the interaction energy of openmm-ml's PyTorch MACE
      potentials, which differs from it by a constant per composition, is not available.

    The Topology must be periodic: the electrostatics are k-space sums over the model's k-vectors (exact, as in
    PyTorch), whose cost grows as the number of atoms times the box volume.
    """

    def __init__(self, name, modelPath):
        if modelPath is None:
            raise ValueError(
                "symmetrix-mace-polar needs modelPath: a Symmetrix JSON or a PyTorch PolarMACE model"
            )
        self.name = name
        self.modelPath = str(modelPath)

    def getMLLongRange(self):
        return True

    def addForces(
        self,
        topology,
        system,
        atoms,
        forceGroup,
        charge=0,
        multiplicity=1,
        device=None,
        precision=None,
        dtype=None,
        returnEnergyType="energy",
        **args,
    ):
        if returnEnergyType != "energy":
            raise ValueError(
                "symmetrix-mace-polar computes only returnEnergyType='energy' (the total energy): model files store "
                "the isolated-atom energies together with the energy shift"
            )
        if (
            topology.getPeriodicBoxVectors() is None
            and not system.usesPeriodicBoundaryConditions()
        ):
            raise ValueError(
                "symmetrix-mace-polar needs a periodic Topology: its electrostatics are k-space sums"
            )
        if any(atom.element is None for atom in topology.atoms()):
            raise ValueError("All atoms in the Topology must have elements defined.")
        included = list(topology.atoms())
        if atoms is not None:
            included = [included[i] for i in atoms]
        numbers = [atom.element.atomic_number for atom in included]

        if precision not in (None, "single", "double", "mixed"):
            raise ValueError(
                f"precision must be 'single', 'double' or 'mixed', not '{precision}'"
            )
        if dtype is None:
            dtype = {
                None: None,
                "single": "float32",
                "double": "float64",
                "mixed": "float32-mixed",
            }[precision]
        if device is None:
            device = "cpu" if dtype == "float64" or not _cuda_built() else "cuda"
        device = str(device)
        if device.startswith("cuda:") and device != "cuda:0":
            raise ValueError(
                "symmetrix-mace-polar runs on the current CUDA device: select it with CUDA_VISIBLE_DEVICES"
            )
        device = device.split(":")[0]

        force = create_mace_polar_force(
            self._model_file(numbers),
            numbers,
            device,
            dtype,
            charge,
            multiplicity,
            atoms,
        )
        force.setForceGroup(forceGroup)
        system.addForce(force)

    def _model_file(self, numbers):
        if self.modelPath.endswith(".json"):
            return self.modelPath
        import json

        from . import __version__
        from .extract_mace_data import extract_mace_data

        elements = sorted(set(numbers))
        st = os.stat(self.modelPath)
        key = f"{os.path.abspath(self.modelPath)}:{st.st_size}:{st.st_mtime_ns}:{elements}:{__version__}"
        name = os.path.splitext(os.path.basename(self.modelPath))[0]
        digest = hashlib.sha256(key.encode()).hexdigest()[:12]
        path = os.path.join(self._getCacheDir(), f"symmetrix-{name}-{digest}.json")
        if not os.path.isfile(path):
            data = extract_mace_data(self.modelPath, elements)
            tmp = f"{path}.{os.getpid()}.tmp"
            with open(tmp, "w") as f:
                json.dump(data, f)
            os.replace(tmp, path)
        return path
