import numpy as np
import openmm.unit as unit

from openmm import *
from openmm.app import *

from abc import abstractmethod, ABC
from scipy.constants import physical_constants


class BaseDynamics(ABC):
    def __init__(self, cfg, state):
        super().__init__()
        self.cfg = cfg
        molecule = cfg.data.molecule
        self.start_file = f"./data/{molecule}/{state}.pdb"

        self.temperature = cfg.steeredmd.simulation.temperature * unit.kelvin
        self.friction = cfg.steeredmd.simulation.friction / unit.femtoseconds
        self.timestep = cfg.steeredmd.simulation.timestep * unit.femtoseconds

        self.pdb, self.integrator, self.simulation, self.external_force = self.setup()

        self.get_md_info()
        self.simulation.minimizeEnergy()
        self.position = self.report()[0]

    @abstractmethod
    def setup(self):
        pass

    def get_md_info(self):
        self.num_particles = self.simulation.system.getNumParticles()
        m = np.array(
            [
                self.simulation.system.getParticleMass(i).value_in_unit(unit.dalton)
                for i in range(self.num_particles)
            ]
        )
        self.heavy_atoms = m > 1.1
        m = unit.Quantity(m, unit.dalton)

        unadjusted_variance = (
            2
            * self.timestep
            * self.friction
            * unit.BOLTZMANN_CONSTANT_kB
            * self.temperature
            / m[:, None]
        )
        std_SI_units = (
            1
            / physical_constants["unified atomic mass unit"][0]
            * unadjusted_variance.value_in_unit(unit.joule / unit.dalton)
        )
        self.std = unit.Quantity(
            np.sqrt(std_SI_units), unit.meter / unit.second
        ).value_in_unit(unit.nanometer / unit.femtosecond)
        self.m = m.value_in_unit(unit.dalton)

    def step(self, forces):
        for i in range(forces.shape[0]):
            self.external_force.setParticleParameters(i, i, forces[i])
        self.external_force.updateParametersInContext(self.simulation.context)
        self.simulation.step(1)

    def report(self):
        state = self.simulation.context.getState(getPositions=True, getForces=True)
        positions = state.getPositions().value_in_unit(unit.nanometer)
        forces = state.getForces().value_in_unit(
            unit.dalton * unit.nanometer / unit.femtosecond**2
        )
        return positions, forces

    def reset(self):
        for i in range(len(self.position)):
            self.external_force.setParticleParameters(i, i, [0, 0, 0])
        self.external_force.updateParametersInContext(self.simulation.context)
        self.simulation.context.setPositions(self.position)
        self.simulation.context.setVelocitiesToTemperature(self.temperature)

    def set_position(self, positions):
        atom_xyz = positions.detach().cpu().numpy()
        atom_list = [Vec3(atom[0], atom[1], atom[2]) for atom in atom_xyz]
        current_state_openmm = unit.Quantity(value=atom_list, unit=unit.nanometer)
        self.simulation.context.setPositions(current_state_openmm)
        self.simulation.context.setVelocitiesToTemperature(self.temperature)

    def set_temperature(self, temperature):
        self.integrator.setTemperature(temperature * unit.kelvin)

    def energy_function(self, positions):
        forces, potentials = [], []
        for i in range(len(positions)):
            self.simulation.context.setPositions(positions[i])
            state = self.simulation.context.getState(getForces=True, getEnergy=True)
            force = state.getForces().value_in_unit(
                unit.dalton * unit.nanometer / unit.femtosecond**2
            )
            potential = state.getPotentialEnergy().value_in_unit(
                unit.kilojoules / unit.mole
            )
            forces.append(force)
            potentials.append(potential)
        return np.array(forces), np.array(potentials)
    
    

def load_forcefield(cfg, molecule):
    if molecule == "alanine":
        forcefield = app.ForceField(*cfg.steeredmd.simulation.force_field)
    elif molecule == "chignolin":
        path = os.path.join(
            os.getcwd(),
            "openmmforcefields/openmmforcefields/ffxml/amber/protein.ff14SBonlysc.xml",
        )
        forcefield = app.ForceField(*cfg.steeredmd.simulation.force_field)
    else:
        raise ValueError(f"Molecule {molecule} not found")
        
    return forcefield

def load_system(cfg, molecule, pdb, forcefield):
    if molecule == "alanine":
        system = forcefield.createSystem(
            pdb.topology,
            nonbondedMethod=app.PME,
            constraints=app.HBonds,
            ewaldErrorTolerance=0.0005,
        )
    elif molecule == "chignolin": 
        system = forcefield.createSystem(
            pdb.topology,
            constraints=app.HBonds,
            ewaldErrorTolerance=0.0005,
        )
    else:
        raise ValueError(f"Molecule {molecule} not found")  
    
    return system