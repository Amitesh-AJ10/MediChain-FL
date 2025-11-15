"""
Automated tests for HEManager encryption, serialization, and aggregation.

Validates:
- Single-layer encryption/decryption round-trip
- Multi-layer gradient handling
- Weighted aggregation with non-uniform weights
- Serialization/deserialization for transport
- Edge cases (empty, single client, large tensors)
"""

import numpy as np
import pytest
from utils.encryption import HEManager


class TestHEManagerRoundTrip:
    """Test basic encryption/decryption cycle."""

    def test_single_layer_roundtrip(self):
        """Encrypt and decrypt a single gradient layer."""
        he = HEManager()
        plain = [np.array([1.0, 2.0, 3.0, 4.0])]
        
        enc = he.encrypt_gradients(plain, encrypt_all=True)
        dec = he.decrypt_gradients(enc)
        
        np.testing.assert_allclose(plain[0], dec[0], rtol=1e-5, atol=1e-6)

    def test_multi_layer_roundtrip(self):
        """Encrypt and decrypt multiple gradient layers."""
        he = HEManager()
        plain = [
            np.random.randn(128).astype(np.float64),
            np.random.randn(64).astype(np.float64),
            np.random.randn(32).astype(np.float64),
        ]
        
        enc = he.encrypt_gradients(plain, encrypt_all=True)
        dec = he.decrypt_gradients(enc)
        
        for orig, recovered in zip(plain, dec):
            np.testing.assert_allclose(orig, recovered, rtol=1e-5, atol=1e-6)

    def test_large_tensor_precision(self):
        """Validate precision on large tensors (1024 elements)."""
        he = HEManager()
        plain = [np.random.randn(1024).astype(np.float64)]
        
        enc = he.encrypt_gradients(plain, encrypt_all=True)
        dec = he.decrypt_gradients(enc)
        
        max_err = np.abs(plain[0] - dec[0]).max()
        assert max_err < 1e-5, f"Max error {max_err} exceeds threshold"


class TestSerialization:
    """Test serialization/deserialization for transport."""

    def test_serialize_deserialize_roundtrip(self):
        """Serialize and deserialize encrypted vectors."""
        he = HEManager()
        plain = [np.array([10.0, 20.0, 30.0])]
        
        enc = he.encrypt_gradients(plain, encrypt_all=True)
        serialized = he.serialize_vectors(enc)
        deserialized = he.deserialize_vectors(serialized)
        dec = he.decrypt_gradients(deserialized)
        
        np.testing.assert_allclose(plain[0], dec[0], rtol=1e-5, atol=1e-6)

    def test_serialized_dtype(self):
        """Ensure serialized vectors are uint8 numpy arrays."""
        he = HEManager()
        plain = [np.array([1.0, 2.0])]
        
        enc = he.encrypt_gradients(plain, encrypt_all=True)
        serialized = he.serialize_vectors(enc)
        
        assert isinstance(serialized, list)
        assert all(isinstance(arr, np.ndarray) for arr in serialized)
        assert all(arr.dtype == np.uint8 for arr in serialized)


class TestWeightedAggregation:
    """Test homomorphic aggregation with custom weights."""

    def test_uniform_weights(self):
        """Aggregate two identical updates with equal weights."""
        he = HEManager()
        plain = [np.array([4.0, 8.0, 12.0])]
        
        enc1 = he.encrypt_gradients(plain, encrypt_all=True)
        enc2 = he.encrypt_gradients(plain, encrypt_all=True)
        
        agg = he.aggregate_encrypted_weighted([enc1, enc2], [1.0, 1.0])
        dec = he.decrypt_gradients(agg)
        
        # With equal weights, result should match original
        np.testing.assert_allclose(plain[0], dec[0], rtol=1e-5, atol=1e-6)

    def test_non_uniform_weights(self):
        """Aggregate with non-uniform weights (e.g., by sample count)."""
        he = HEManager()
        client1_grads = [np.array([2.0, 4.0, 6.0])]
        client2_grads = [np.array([8.0, 10.0, 12.0])]
        
        enc1 = he.encrypt_gradients(client1_grads, encrypt_all=True)
        enc2 = he.encrypt_gradients(client2_grads, encrypt_all=True)
        
        # Weight by sample counts: client1=100, client2=300
        agg = he.aggregate_encrypted_weighted([enc1, enc2], [100, 300])
        dec = he.decrypt_gradients(agg)
        
        # Expected: (2*100 + 8*300)/400 = 6.5, (4*100 + 10*300)/400 = 8.5, ...
        expected = np.array([6.5, 8.5, 10.5])
        np.testing.assert_allclose(expected, dec[0], rtol=1e-5, atol=1e-6)

    def test_three_clients_aggregation(self):
        """Aggregate gradients from three clients."""
        he = HEManager()
        grads = [
            [np.array([1.0, 2.0])],
            [np.array([3.0, 4.0])],
            [np.array([5.0, 6.0])],
        ]
        
        encrypted = [he.encrypt_gradients(g, encrypt_all=True) for g in grads]
        agg = he.aggregate_encrypted_weighted(encrypted, [1, 1, 1])
        dec = he.decrypt_gradients(agg)
        
        # Expected: mean([1,3,5], [2,4,6]) = [3.0, 4.0]
        expected = np.array([3.0, 4.0])
        np.testing.assert_allclose(expected, dec[0], rtol=1e-5, atol=1e-6)


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_single_client_aggregation(self):
        """Aggregate a single client (should return the same gradient)."""
        he = HEManager()
        plain = [np.array([7.0, 14.0, 21.0])]
        
        enc = he.encrypt_gradients(plain, encrypt_all=True)
        agg = he.aggregate_encrypted_weighted([enc], [1.0])
        dec = he.decrypt_gradients(agg)
        
        np.testing.assert_allclose(plain[0], dec[0], rtol=1e-5, atol=1e-6)

    def test_zero_weight_raises_error(self):
        """Zero total weight should raise ValueError."""
        he = HEManager()
        plain = [np.array([1.0])]
        enc = he.encrypt_gradients(plain, encrypt_all=True)
        
        with pytest.raises(ValueError, match="Total weight must be positive"):
            he.aggregate_encrypted_weighted([enc], [0.0])

    def test_mismatched_weights_raises_error(self):
        """Mismatched number of weights and clients should raise ValueError."""
        he = HEManager()
        plain = [np.array([1.0])]
        enc = he.encrypt_gradients(plain, encrypt_all=True)
        
        with pytest.raises(ValueError, match="Weights length must match"):
            he.aggregate_encrypted_weighted([enc, enc], [1.0])


class TestContextSerialization:
    """Test HE context sharing for client/server setup."""

    def test_context_serialization(self):
        """Server can serialize context for distribution to clients."""
        server_he = HEManager()
        context_bytes = server_he.serialize_context()
        
        assert isinstance(context_bytes, bytes)
        assert len(context_bytes) > 0

    def test_client_from_serialized_context(self):
        """Client can reconstruct HE manager from serialized context."""
        server_he = HEManager()
        context_bytes = server_he.serialize_context()
        
        client_he = HEManager.from_serialized(context_bytes, has_secret=False)
        
        # Client encrypts with public context
        plain = [np.array([5.0, 10.0])]
        enc = client_he.encrypt_gradients(plain, encrypt_all=True)
        
        # Serialize encrypted vectors for transport
        serialized = client_he.serialize_vectors(enc)
        
        # Server deserializes with its own (secret) context and decrypts
        deserialized = server_he.deserialize_vectors(serialized)
        dec = server_he.decrypt_gradients(deserialized)
        np.testing.assert_allclose(plain[0], dec[0], rtol=1e-5, atol=1e-6)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
