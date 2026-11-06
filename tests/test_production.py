import unittest
import time
import os
import torch
from apps.api.services.simulation_queue import SimulationQueue
from models.onnx_exporter import ProductionExporter


class TestProductionInfrastructure(unittest.TestCase):
    def test_simulation_queue_async(self):
        queue = SimulationQueue(max_workers=2)

        def mock_sim(x):
            time.sleep(0.05)
            return {"sim_result": x * 2}

        job_id = queue.submit_job(mock_sim, x=21)
        self.assertIsNotNone(job_id)

        job = queue.get_job(job_id)
        self.assertIn(job.status, ["queued", "running", "completed"])

        # Wait briefly for completion
        time.sleep(0.15)
        completed_job = queue.get_job(job_id)
        self.assertEqual(completed_job.status, "completed")
        self.assertEqual(completed_job.result["sim_result"], 42)

    def test_production_exporter_torchscript(self):
        exporter = ProductionExporter(num_genes=50, latent_dim=16, export_dir="artifacts/test_export")
        enc, gnn = exporter.load_trained_models()

        jit_paths = exporter.export_torchscript(enc, gnn, sample_batch_size=2)
        self.assertTrue(os.path.exists(jit_paths["encoder"]))
        self.assertTrue(os.path.exists(jit_paths["decoder"]))

        parity = exporter.verify_numerical_parity(enc, jit_paths["encoder"], n_samples=4)
        self.assertLess(parity, 1e-4)

        # Cleanup test artifacts
        for p in jit_paths.values():
            if os.path.exists(p):
                os.remove(p)
        if os.path.exists("artifacts/test_export"):
            try:
                os.rmdir("artifacts/test_export")
            except Exception:
                pass


if __name__ == "__main__":
    unittest.main()
