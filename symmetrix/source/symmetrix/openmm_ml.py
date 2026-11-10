"""Symmetrix models in OpenMM, through an openmm.PythonForce.

With openmm-ml installed, the potentials are registered as "symmetrix-mace-polar" (the models of
pair_style symmetrix/mace/polar), "symmetrix-mace-foundation" and "symmetrix-nequip-foundation" (those of
symmetrix/mace/foundation and symmetrix/nequip/foundation):

    from openmmml import MLPotential
    potential = MLPotential("symmetrix-mace-polar", modelPath="polar-m-H-O.json")
    system = potential.createSystem(topology, charge=0, multiplicity=1)

Without openmm-ml, create_mace_polar_force(), create_mace_foundation_force() and create_nequip_foundation_force()
return the Force to add to a System.
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


_FOUNDATION_EVALUATORS = {"mace": "_mace_foundation", "nequip": "_nequip_foundation"}


class FoundationComputation:
    """The computation function of the PythonForce of a model of pair_style symmetrix/mace/foundation or
    symmetrix/nequip/foundation, evaluated by the evaluator of symmetrix.experimental.MACEFoundation or
    NequIPFoundation, which keeps its own neighbor list. It pickles as its arguments, like MACEPolarComputation."""

    def __init__(
        self,
        kind,
        model_file,
        atomic_numbers,
        device="cuda",
        dtype="float64",
        skin=1.0,
        periodic=True,
    ):
        if kind not in _FOUNDATION_EVALUATORS:
            raise ValueError(f"kind must be 'mace' or 'nequip', not '{kind}'")
        self.args = {
            "kind": kind,
            "model_file": str(model_file),
            "atomic_numbers": [int(z) for z in atomic_numbers],
            "device": device,
            "dtype": dtype,
            "skin": float(skin),
            "periodic": bool(periodic),
        }
        self.evaluator = None

    def setup(self):
        import importlib

        a = self.args
        module = importlib.import_module(
            f".{_FOUNDATION_EVALUATORS[a['kind']]}", __package__
        )
        Evaluator = getattr(
            module, f"{'MACE' if a['kind'] == 'mace' else 'NequIP'}Foundation"
        )
        self.evaluator = Evaluator(a["model_file"], a["device"], a["dtype"], a["skin"])
        self.numbers = np.asarray(a["atomic_numbers"], dtype=np.int32)
        self.pbc = np.full(3, a["periodic"])

    def __getstate__(self):
        return self.args

    def __setstate__(self, args):
        self.args = args
        self.evaluator = None

    def __call__(self, state):
        if self.evaluator is None:
            self.setup()
        x = state.getPositions(asNumpy=True).value_in_unit(unit.angstrom)
        if self.args["periodic"]:
            cell = state.getPeriodicBoxVectors(asNumpy=True).value_in_unit(
                unit.angstrom
            )
        else:
            cell = np.zeros((3, 3))
        energy, forces, _, _ = self.evaluator.compute(x, self.numbers, cell, self.pbc)
        return energy * EV_TO_KJ_PER_MOL, forces * (10 * EV_TO_KJ_PER_MOL)


def _create_foundation_force(
    kind, model_file, atomic_numbers, device, dtype, skin, periodic, particles
):
    compute = FoundationComputation(
        kind, model_file, atomic_numbers, device, dtype, skin, periodic
    )
    compute.setup()  # fails here, not in the first step, on a bad model, device or element
    force = openmm.PythonForce(compute)
    force.setUsesPeriodicBoundaryConditions(bool(periodic))
    if particles is not None:
        force.setParticles(list(particles))
    force.setName(f"Symmetrix{'MACE' if kind == 'mace' else 'NequIP'}Foundation")
    return force


def create_mace_foundation_force(
    model_file,
    atomic_numbers,
    device="cuda",
    dtype="float64",
    skin=1.0,
    periodic=True,
    particles=None,
):
    """An openmm.PythonForce evaluating a model of pair_style symmetrix/mace/foundation (Symmetrix JSON) as
    symmetrix.experimental.MACEFoundation does, with a neighbor list built again when an atom has moved by half of
    `skin` (A). periodic=False is an isolated molecule.

    atomic_numbers are those of the particles it applies to: all of the System's, or those in `particles`. The energy
    is the model's total energy, isolated-atom energies included."""
    return _create_foundation_force(
        "mace", model_file, atomic_numbers, device, dtype, skin, periodic, particles
    )


def create_nequip_foundation_force(
    model_file,
    atomic_numbers,
    device="cuda",
    dtype="float64",
    skin=1.0,
    periodic=True,
    particles=None,
):
    """As create_mace_foundation_force(), for a model of pair_style symmetrix/nequip/foundation, evaluated as
    symmetrix.experimental.NequIPFoundation does."""
    return _create_foundation_force(
        "nequip", model_file, atomic_numbers, device, dtype, skin, periodic, particles
    )


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


def _atomic_numbers(topology, atoms):
    if any(atom.element is None for atom in topology.atoms()):
        raise ValueError("All atoms in the Topology must have elements defined.")
    included = list(topology.atoms())
    if atoms is not None:
        included = [included[i] for i in atoms]
    return [atom.element.atomic_number for atom in included]


def _dtype(precision, dtype):
    if precision not in (None, "single", "double", "mixed"):
        raise ValueError(
            f"precision must be 'single', 'double' or 'mixed', not '{precision}'"
        )
    if dtype is not None:
        return dtype
    return {
        None: None,
        "single": "float32",
        "double": "float64",
        "mixed": "float32-mixed",
    }[precision]


def _device(name, device):
    device = str(device)
    if device.startswith("cuda:") and device != "cuda:0":
        raise ValueError(
            f"{name} runs on the current CUDA device: select it with CUDA_VISIBLE_DEVICES"
        )
    return device.split(":")[0]


def _is_periodic(topology, system):
    return (
        topology.getPeriodicBoxVectors() is not None
        or system.usesPeriodicBoundaryConditions()
    )


class _SymmetrixImpl(MLPotentialImpl):
    """What the openmm-ml potentials of Symmetrix share: modelPath, a Symmetrix JSON or a PyTorch model, which is then
    extracted for the elements of the Topology into openmm-ml's cache directory."""

    def __init__(self, name, modelPath):
        if modelPath is None:
            raise ValueError(
                f"{name} needs modelPath: a Symmetrix JSON or a model to extract"
            )
        self.name = name
        self.modelPath = str(modelPath)

    def _check_energy_type(self, returnEnergyType):
        if returnEnergyType != "energy":
            raise ValueError(
                f"{self.name} computes only returnEnergyType='energy' (the total energy): model files store "
                "the isolated-atom energies together with the energy shift"
            )

    def _extract(self, elements):
        from .extract_mace_data import extract_mace_data

        return extract_mace_data(self.modelPath, elements)

    def _model_file(self, numbers):
        if self.modelPath.endswith(".json"):
            return self.modelPath
        import json

        from . import __version__

        elements = sorted(set(numbers))
        if os.path.exists(self.modelPath):
            st = os.stat(self.modelPath)
            source = f"{os.path.abspath(self.modelPath)}:{st.st_size}:{st.st_mtime_ns}"
        else:  # a model identifier, such as a nequip.net one
            source = self.modelPath
        key = f"{self.name}:{source}:{elements}:{__version__}"
        name = os.path.splitext(os.path.basename(self.modelPath.replace(":", "-")))[0]
        digest = hashlib.sha256(key.encode()).hexdigest()[:12]
        path = os.path.join(self._getCacheDir(), f"symmetrix-{name}-{digest}.json")
        if not os.path.isfile(path):
            data = self._extract(elements)
            tmp = f"{path}.{os.getpid()}.tmp"
            with open(tmp, "w") as f:
                json.dump(data, f)
            os.replace(tmp, path)
        return path


class SymmetrixMACEPolarImpl(_SymmetrixImpl):
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
        self._check_energy_type(returnEnergyType)
        if not _is_periodic(topology, system):
            raise ValueError(
                f"{self.name} needs a periodic Topology: its electrostatics are k-space sums"
            )
        numbers = _atomic_numbers(topology, atoms)
        dtype = _dtype(precision, dtype)
        if device is None:
            device = "cpu" if dtype == "float64" or not _cuda_built() else "cuda"
        force = create_mace_polar_force(
            self._model_file(numbers),
            numbers,
            _device(self.name, device),
            dtype,
            charge,
            multiplicity,
            atoms,
        )
        force.setForceGroup(forceGroup)
        system.addForce(force)


class SymmetrixFoundationImpl(_SymmetrixImpl):
    """Models of pair_style symmetrix/mace/foundation ("symmetrix-mace-foundation") or symmetrix/nequip/foundation
    ("symmetrix-nequip-foundation") evaluated by Symmetrix, for openmm-ml, where symmetrix_openmmml registers them.

    >>> potential = MLPotential("symmetrix-mace-foundation", modelPath="mace-omat-0-small-H-O.json")
    >>> system = potential.createSystem(topology, device="cuda", precision="single")

    modelPath is a Symmetrix JSON, or a model that symmetrix_extract_mace or symmetrix_extract_nequip accepts, which
    is then extracted for the elements of the Topology (with the extraction's dependencies) into openmm-ml's cache
    directory.

    Arguments of createSystem() and createMixedSystem():

    - device: "cuda" (the default when Symmetrix was built with CUDA) or "cpu". The GPU is the current CUDA device.
    - precision: "double" (the default), "single" or "mixed" (GPU, fp16 tensor-core GEMMs to float32 accuracy); or
      dtype as for symmetrix.experimental.MACEFoundation.
    - skin: the neighbor list is built again when an atom has moved by half of it (A, default 1).
    - returnEnergyType: only "energy", the total energy with the isolated-atom energies, as for
      symmetrix-mace-polar.

    A Topology with periodic box vectors is periodic in all three directions; one without is an isolated molecule.
    """

    def __init__(self, name, modelPath, kind):
        super().__init__(name, modelPath)
        self.kind = kind

    def _extract(self, elements):
        if self.kind == "mace":
            return super()._extract(elements)
        from .extract_nequip_data import extract_nequip_data

        return extract_nequip_data(self.modelPath, elements)

    def addForces(
        self,
        topology,
        system,
        atoms,
        forceGroup,
        device=None,
        precision=None,
        dtype=None,
        skin=1.0,
        returnEnergyType="energy",
        **args,
    ):
        self._check_energy_type(returnEnergyType)
        numbers = _atomic_numbers(topology, atoms)
        dtype = _dtype(precision, dtype) or "float64"
        model_file = self._model_file(numbers)
        periodic = _is_periodic(topology, system)

        def create(device):
            return _create_foundation_force(
                self.kind, model_file, numbers, device, dtype, skin, periodic, atoms
            )

        if device is not None:
            force = create(_device(self.name, device))
        else:
            try:
                force = create("cuda")
            except RuntimeError as e:
                if "built without CUDA" not in str(e):
                    raise
                force = create("cpu")
        force.setForceGroup(forceGroup)
        system.addForce(force)
