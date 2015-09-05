"""
Unittests for serializing various objects in ParmEd
"""
from __future__ import division

from io import BytesIO
import numpy as np
import parmed as pmd
from parmed.utils.six.moves import range, zip
try:
    raise ImportError()
    import cPickle as pickle
except ImportError:
    import pickle
import random
try:
    from string import uppercase
except ImportError:
    from string import ascii_uppercase as uppercase
import unittest
import utils

class TestParmedSerialization(unittest.TestCase):
    """ Tests ParmEd serialization """

    def _equal_atoms(self, a1, a2):
        self.assertEqual(a1.atomic_number, a2.atomic_number)
        self.assertEqual(a1.screen, a2.screen)
        self.assertEqual(a1.name, a2.name)
        self.assertEqual(a1.type, a2.type)
        self.assertEqual(a1.atom_type, a2.atom_type)
        self.assertEqual(a1.charge, a2.charge)
        self.assertEqual(a1.mass, a2.mass)
        self.assertEqual(a1.nb_idx, a2.nb_idx)
        self.assertEqual(a1.radii, a2.radii)
        self.assertEqual(a1.tree, a2.tree)
        self.assertEqual(a1.join, a2.join)
        self.assertEqual(a1.irotat, a2.irotat)
        self.assertEqual(a1.occupancy, a2.occupancy)
        self.assertEqual(a1.bfactor, a2.bfactor)
        self.assertEqual(a1.rmin, a2.rmin)
        self.assertEqual(a1.epsilon, a2.epsilon)
        self.assertEqual(a1.rmin_14, a2.rmin_14)
        self.assertEqual(a1.epsilon_14, a2.epsilon_14)
        for key in ('xx', 'xy', 'xz', 'vx', 'vy', 'vz', 'multipoles',
                    'type_idx', 'class_idx', 'polarizability', 'vdw_weight',
                    'segid'):
            if hasattr(a2, key):
                if isinstance(getattr(a2, key), np.ndarray):
                    np.testing.assert_equal(
                            getattr(a1, key), getattr(a2, key)
                    )
                else:
                    self.assertEqual(getattr(a1, key), getattr(a2, key))
            else:
                self.assertFalse(hasattr(a1, key))

    def test_atom_serialization(self):
        """ Tests serialization/pickleability of Atom """
        atom = pmd.Atom(atomic_number=random.randint(1, 100),
                        name=random.choice(uppercase)+random.choice(uppercase),
                        type=random.choice(uppercase)+random.choice(uppercase),
                        charge=random.random()*2-1, mass=random.random()*30+1,
                        nb_idx=random.randint(1, 20), radii=random.random()*2,
                        screen=random.random()*2, tree='M',
                        join=random.random()*2, irotat=random.random(),
                        occupancy=random.random(), bfactor=random.random()*10,
                        altloc=random.choice(uppercase), rmin=random.random()*2,
                        epsilon=random.random()/2, rmin14=random.random()*2,
                        epsilon14=random.random()/2)
        atom.xx, atom.xy, atom.xz = (random.random()*100-50 for i in range(3))
        atom.number = random.randint(1, 100)
        atom.vx, atom.vy, atom.vz = (random.random()*100-50 for i in range(3))
        atom.multipoles = np.random.rand(10) * 10

        fobj = BytesIO()
        pickle.dump(atom, fobj)
        fobj.seek(0)
        unpickled = pickle.load(fobj)
        
        self.assertIsInstance(unpickled, pmd.Atom)
        self._equal_atoms(unpickled, atom)

    def test_bond_serialization(self):
        """ Tests serialization/pickleability of Bond """
        struct = utils.create_random_structure(True)
        bond = struct.bonds[0]
        fobj = BytesIO()
        pickle.dump(bond, fobj)
        fobj.seek(0)
        unpickled = pickle.load(fobj)

        self.assertIsInstance(bond, pmd.Bond)

    def test_bondtype_serialization(self):
        """ Tests serialization/pickleability of BondType """
        struct = utils.create_random_structure(True)
        bt = struct.bond_types[0]

        fobj = BytesIO()
        pickle.dump(bt, fobj)
        fobj.seek(0)

        unpickled = pickle.load(fobj)

        self.assertEqual(unpickled, bt)
        self.assertIsNot(unpickled, bt)

    def test_residue_serialization(self):
        """ Tests serialization/pickleability of Residue """
        struct = utils.create_random_structure(parametrized=True)
        res = struct.residues[0]

        fobj = BytesIO()
        pickle.dump(res, fobj)
        fobj.seek(0)
        unpickled = pickle.load(fobj)

        self.assertEqual(len(res.atoms), len(unpickled.atoms))
        for a1, a2 in zip(res, unpickled):
            self._equal_atoms(a1, a2)

    def _compare_structures(self, unpickled, structure):
        for a1, a2 in zip(unpickled, structure):
            self._equal_atoms(a1, a2)
            self.assertEqual(a1.idx, a2.idx)

        # Make sure all of the type arrays are equivalent
        def cmp_type_arrays(arr1, arr2):
            self.assertEqual(len(arr1), len(arr2))
            for x1, x2 in zip(arr1, arr2):
                self.assertEqual(x1, x2)

        cmp_type_arrays(structure.bond_types, unpickled.bond_types)
        cmp_type_arrays(structure.angle_types, unpickled.angle_types)
        cmp_type_arrays(structure.dihedral_types, unpickled.dihedral_types)
        cmp_type_arrays(structure.improper_types, unpickled.improper_types)
        cmp_type_arrays(structure.urey_bradley_types, unpickled.urey_bradley_types)
        cmp_type_arrays(structure.rb_torsion_types, unpickled.rb_torsion_types)
        cmp_type_arrays(structure.cmap_types, unpickled.cmap_types)
        cmp_type_arrays(structure.trigonal_angle_types,
                        unpickled.trigonal_angle_types)
        cmp_type_arrays(structure.out_of_plane_bend_types,
                        unpickled.out_of_plane_bend_types)
        cmp_type_arrays(structure.stretch_bend_types, unpickled.stretch_bend_types)
        cmp_type_arrays(structure.torsion_torsion_types,
                        unpickled.torsion_torsion_types)
        cmp_type_arrays(structure.pi_torsion_types, unpickled.pi_torsion_types)
        cmp_type_arrays(structure.adjust_types, unpickled.adjust_types)
        cmp_type_arrays(structure.groups, unpickled.groups)

        # Make sure all of the connectivity arrays are equivalent
        def cmp_top_arrays(arr1, arr2):
            self.assertEqual(len(arr1), len(arr2))
            for t1, t2 in zip(arr1, arr2):
                self.assertIs(type(t1), type(t2))
                atoms = [attr for attr in dir(t1) if attr.startswith('atom')]
                for a in atoms:
                    self._equal_atoms(getattr(t1, a), getattr(t2, a))
                if hasattr(t1, 'type'):
                    self.assertEqual(t1.type, t2.type)

        cmp_top_arrays(structure.bonds, unpickled.bonds)
        cmp_top_arrays(structure.angles, unpickled.angles)
        cmp_top_arrays(structure.dihedrals, unpickled.dihedrals)
        cmp_top_arrays(structure.impropers, unpickled.impropers)
        cmp_top_arrays(structure.urey_bradleys, unpickled.urey_bradleys)
        cmp_top_arrays(structure.rb_torsions, unpickled.rb_torsions)
        cmp_top_arrays(structure.cmaps, unpickled.cmaps)
        cmp_top_arrays(structure.trigonal_angles, unpickled.trigonal_angles)
        cmp_top_arrays(structure.out_of_plane_bends, unpickled.out_of_plane_bends)
        cmp_top_arrays(structure.pi_torsions, unpickled.pi_torsions)
        cmp_top_arrays(structure.stretch_bends, unpickled.stretch_bends)
        cmp_top_arrays(structure.torsion_torsions, unpickled.torsion_torsions)
        cmp_top_arrays(structure.chiral_frames, unpickled.chiral_frames)
        cmp_top_arrays(structure.multipole_frames, unpickled.multipole_frames)
        cmp_top_arrays(structure.adjusts, unpickled.adjusts)

    def test_structure_serialization(self):
        """ Tests serialization/pickleability of Structure """
        structure = utils.create_random_structure(parametrized=True)
        fobj = BytesIO()
        pickle.dump(structure, fobj)
        fobj.seek(0)
        unpickled = pickle.load(fobj)

        self.assertEqual(len(unpickled.residues), len(structure.residues))
        for r1, r2 in zip(unpickled.residues, structure.residues):
            self.assertEqual(len(r1), len(r2))
            self.assertEqual(r1.idx, r2.idx)
            for a1, a2 in zip(r1, r2):
                self._equal_atoms(a1, a2)
        self._compare_structures(unpickled, structure)

    def test_fortran_format_serialization(self):
        """ Tests the serialization/pickleability of FortranFormat """
        fmt = pmd.amber.FortranFormat('8I10')
        unpickled = pickle.loads(pickle.dumps(fmt))

        self.assertEqual(fmt.format, unpickled.format)
        self.assertEqual(fmt.strip_strings, unpickled.strip_strings)
        self.assertIs(fmt.type, unpickled.type)
        self.assertEqual(fmt.nitems, unpickled.nitems)
        self.assertEqual(fmt.itemlen, unpickled.itemlen)
        self.assertEqual(fmt.fmt, unpickled.fmt)

    def test_amberformat_serialization(self):
        """ Tests serialization/pickleability of AmberFormat """
        amber = pmd.load_file(utils.get_fn('cSPCE.mdl'))
        unpickled = pickle.loads(pickle.dumps(amber))

        self.assertEqual(set(amber.parm_data.keys()),
                         set(unpickled.parm_data.keys()))
        self.assertEqual(amber.flag_list, unpickled.flag_list)
        self.assertEqual(set(amber.formats.keys()),
                         set(unpickled.formats.keys()))
        for k1 in amber.parm_data.keys():
            self.assertEqual(amber.parm_data[k1], unpickled.parm_data[k1])
            self.assertEqual(amber.formats[k1], unpickled.formats[k1])

        self.assertEqual(amber.charge_flag, unpickled.charge_flag)
        self.assertEqual(amber.version, unpickled.version)
        self.assertEqual(amber.name, unpickled.name)

    def test_amberparm_serialization(self):
        """ Tests the serialization/pickleability of AmberParm """
        structure = pmd.load_file(utils.get_fn('ash.parm7'))
        unpickled = pickle.loads(pickle.dumps(structure))

        self._compare_structures(unpickled, structure)

        self.assertEqual(set(structure.parm_data.keys()),
                         set(unpickled.parm_data.keys()))
        self.assertEqual(structure.flag_list, unpickled.flag_list)
        self.assertEqual(set(structure.formats.keys()),
                         set(unpickled.formats.keys()))
        for k1 in structure.parm_data.keys():
            self.assertEqual(structure.parm_data[k1], unpickled.parm_data[k1])
            self.assertEqual(structure.formats[k1], unpickled.formats[k1])

        self.assertEqual(structure.charge_flag, unpickled.charge_flag)
        self.assertEqual(structure.version, unpickled.version)
        self.assertEqual(structure.name, unpickled.name)
        self.assertIs(pmd.amber.AmberParm, type(unpickled))

    def test_chamberparm_serialization(self):
        """ Tests the serialization/pickleability of ChamberParm """
        structure = pmd.load_file(utils.get_fn('ala_ala_ala.parm7'),
                                  utils.get_fn('ala_ala_ala.rst7'))
        unpickled = pickle.loads(pickle.dumps(structure))

        self._compare_structures(unpickled, structure)

        self.assertEqual(set(structure.parm_data.keys()),
                         set(unpickled.parm_data.keys()))
        self.assertEqual(structure.flag_list, unpickled.flag_list)
        self.assertEqual(set(structure.formats.keys()),
                         set(unpickled.formats.keys()))
        for k1 in structure.parm_data.keys():
            self.assertEqual(structure.parm_data[k1], unpickled.parm_data[k1])
            self.assertEqual(structure.formats[k1], unpickled.formats[k1])

        self.assertEqual(structure.charge_flag, unpickled.charge_flag)
        self.assertEqual(structure.version, unpickled.version)
        self.assertEqual(structure.name, unpickled.name)
        self.assertIs(pmd.amber.ChamberParm, type(unpickled))
