import tempfile
import unittest
from pathlib import Path

from msc_project_uav_mas_pso import (
    ALGORITHMS,
    FORMAL_OUTPUT_FILENAMES,
    FORMAL_PSO_CONFIG,
    OBSOLETE_OUTPUT_FILENAMES,
    PSOConfig,
    remove_obsolete_outputs,
    validate_formal_experiment,
    verify_formal_outputs,
)


class FormalReleaseTests(unittest.TestCase):
    def test_formal_parameters_are_fixed(self):
        config = FORMAL_PSO_CONFIG

        self.assertEqual(config.particles, 32)
        self.assertEqual(config.iterations, 25)
        self.assertEqual(config.inertia, 0.72)
        self.assertEqual(config.c1, 1.45)
        self.assertEqual(config.c2, 1.45)
        self.assertEqual(config.min_safe_distance, 12.0)
        self.assertEqual(config.max_move_distance, 35.0)
        self.assertEqual(config.safe_penalty_weight, 0.18)
        self.assertEqual(config.move_penalty_weight, 0.12)
        self.assertEqual(config.overlap_penalty_weight, 0.04)
        self.assertEqual(config.density_bonus_weight, 0.03)
        validate_formal_experiment()

    def test_configuration_defaults_remain_compatible_with_original_code(self):
        config = PSOConfig()
        self.assertEqual(config.particles, 24)
        self.assertEqual(config.iterations, 30)

    def test_output_contract_is_unique_and_has_no_obsolete_names(self):
        self.assertEqual(
            len(FORMAL_OUTPUT_FILENAMES),
            len(set(FORMAL_OUTPUT_FILENAMES)),
        )
        self.assertTrue(
            set(FORMAL_OUTPUT_FILENAMES).isdisjoint(
                OBSOLETE_OUTPUT_FILENAMES
            )
        )

    def test_cleanup_removes_only_known_obsolete_files(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            obsolete = output / OBSOLETE_OUTPUT_FILENAMES[0]
            retained = output / "notes.txt"
            obsolete.write_text("old", encoding="utf-8")
            retained.write_text("keep", encoding="utf-8")

            removed = remove_obsolete_outputs(output)

            self.assertEqual(removed, (obsolete,))
            self.assertFalse(obsolete.exists())
            self.assertTrue(retained.exists())

    def test_output_verification_requires_every_formal_file(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            for filename in FORMAL_OUTPUT_FILENAMES:
                (output / filename).touch()
            verify_formal_outputs(output)

            (output / FORMAL_OUTPUT_FILENAMES[0]).unlink()
            with self.assertRaises(RuntimeError):
                verify_formal_outputs(output)


if __name__ == "__main__":
    unittest.main()
