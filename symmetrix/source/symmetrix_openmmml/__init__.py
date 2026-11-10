"""The openmm-ml entry point of "symmetrix-mace-polar" (symmetrix.openmm_ml). openmm-ml loads every registered
potential when it is imported, so this module imports nothing from symmetrix: a symmetrix that cannot be imported
(missing CUDA libraries, say) fails only when the potential is used."""

from openmmml.mlpotential import MLPotentialImplFactory


class SymmetrixMACEPolarImplFactory(MLPotentialImplFactory):
    """Creates symmetrix.openmm_ml.SymmetrixMACEPolarImpl objects."""

    def createImpl(self, name, modelPath=None, **args):
        from symmetrix.openmm_ml import SymmetrixMACEPolarImpl

        return SymmetrixMACEPolarImpl(name, modelPath)
