"""Model-free roundtrip tests for the documented MoE weight filter."""
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

SMALLTHINKER = Path(__file__).resolve().parents[1]
GGUF_PY = SMALLTHINKER / "gguf-py"
sys.path.insert(0, str(GGUF_PY))

import numpy as np  # noqa: E402
from gguf import GGUFEndian, GGUFReader, GGUFWriter  # noqa: E402


class FilterMoEWeightsTests(unittest.TestCase):
    def check_roundtrip(self, endian, tensor_names):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "input.gguf"
            destination = Path(tmp) / "output.gguf"
            writer = GGUFWriter(source, "smallthinker", endianess=endian)
            writer.add_name("tiny endianness fixture")
            for index, name in enumerate(tensor_names):
                values = np.arange(6, dtype=np.float32).reshape(2, 3) + index
                writer.add_tensor(name, values)
            writer.write_header_to_file()
            writer.write_kv_data_to_file()
            writer.write_tensors_to_file()
            writer.close()

            result = subprocess.run(
                [sys.executable, str(SMALLTHINKER / "get_no_moe_weights_ffn.py"),
                 str(source), str(destination)],
                env=dict(os.environ, PYTHONPATH=str(GGUF_PY)),
                capture_output=True, text=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            # The magic is a fixed byte sequence, independent of the file's
            # integer and tensor byte order (as in GGUFWriter/GGUFReader).
            self.assertEqual(destination.read_bytes()[:4], b"GGUF")

            original = GGUFReader(source)
            converted = GGUFReader(destination)
            try:
                self.assertEqual(converted.endianess, endian)
                self.assertEqual(
                    converted.fields["general.name"].parts[-1].tobytes(),
                    original.fields["general.name"].parts[-1].tobytes(),
                )
                retained = [t for t in original.tensors if "exps" not in t.name]
                self.assertEqual([t.name for t in converted.tensors], [t.name for t in retained])
                for before, after in zip(retained, converted.tensors):
                    np.testing.assert_array_equal(after.shape, before.shape)
                    self.assertEqual(after.tensor_type, before.tensor_type)
                    self.assertEqual(after.data.tobytes(), before.data.tobytes())
                    self.assertEqual(after.data_offset % converted.alignment, 0)
            finally:
                # GGUFReader uses mmap. Release all tensor views before the
                # temporary directory is removed on Windows.
                original.data._mmap.close()
                converted.data._mmap.close()

    def test_little_endian_filter(self):
        self.check_roundtrip(GGUFEndian.LITTLE, ["token_embd.weight", "blk.0.ffn_up_exps.weight"])

    def test_big_endian_filter(self):
        self.check_roundtrip(GGUFEndian.BIG, ["token_embd.weight", "blk.0.ffn_up_exps.weight"])

    def test_little_endian_no_experts(self):
        self.check_roundtrip(GGUFEndian.LITTLE, ["token_embd.weight"])

    def test_big_endian_no_experts(self):
        self.check_roundtrip(GGUFEndian.BIG, ["token_embd.weight"])

    def test_little_endian_only_experts(self):
        self.check_roundtrip(GGUFEndian.LITTLE, ["blk.0.ffn_up_exps.weight"])

    def test_big_endian_only_experts(self):
        self.check_roundtrip(GGUFEndian.BIG, ["blk.0.ffn_up_exps.weight"])


if __name__ == "__main__":
    unittest.main()
