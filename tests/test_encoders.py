import unittest
import torch
from models.encoders.transcriptomic_encoder import TranscriptomicEncoder
from models.encoders.multimodal_encoder import MultimodalEncoder
from models.encoders.chemical_encoder import ChemicalEncoder, REFERENCE_DRUG_REGISTRY


class TestEncoders(unittest.TestCase):
    def test_transcriptomic_encoder(self):
        enc = TranscriptomicEncoder(num_genes=50, latent_dim=16)
        x = torch.randn(4, 50)
        z = enc(x)
        self.assertEqual(z.shape, (4, 50, 16))

    def test_multimodal_cross_attention(self):
        m_enc = MultimodalEncoder(
            num_genes=50,
            n_epigenomic_feats=1,
            latent_dim=16,
            fusion_strategy="cross_attention",
        )
        x_rna = torch.randn(2, 50)
        x_atac = torch.randn(2, 50, 1)

        # Dual modality
        z_both = m_enc(x_rna, x_atac)
        self.assertEqual(z_both.shape, (2, 50, 16))

        # Graceful single-modality fallback
        z_rna_only = m_enc(x_rna)
        self.assertEqual(z_rna_only.shape, (2, 50, 16))

    def test_multimodal_hybrid(self):
        m_enc = MultimodalEncoder(
            num_genes=50,
            latent_dim=16,
            fusion_strategy="hybrid",
        )
        x_rna = torch.randn(2, 50)
        x_atac = torch.randn(2, 50, 1)
        z = m_enc(x_rna, x_atac)
        self.assertEqual(z.shape, (2, 50, 16))

    def test_chemical_encoder(self):
        chem_enc = ChemicalEncoder(num_genes=50, fp_dim=512, latent_dim=16)
        
        # Test reference drug encoding
        latent, pert, affinity = chem_enc.encode_reference_drug("Vorinostat")
        self.assertEqual(latent.shape, (1, 16))
        self.assertEqual(pert.shape, (1, 50))
        self.assertEqual(affinity.shape, (1, 50))
        self.assertTrue((affinity >= 0.0).all() and (affinity <= 1.0).all())


if __name__ == "__main__":
    unittest.main()
