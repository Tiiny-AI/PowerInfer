# -*- coding: utf-8 -*-
from __future__ import annotations
import argparse
import struct
import sys
from pathlib import Path


try:
    from gguf.gguf_reader import GGUFReader, ReaderTensor
    from gguf.constants import GGUF_MAGIC, GGUFEndian
except ImportError:
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from gguf.gguf_reader import GGUFReader, ReaderTensor
    from gguf.constants import GGUF_MAGIC, GGUFEndian


def looks_like_moe(name: str) -> bool:
    return 'exps' in name


def align_offset(offset: int, alignment: int) -> int:
    if alignment == 0:
        return offset
    return offset + (alignment - (offset % alignment)) % alignment


def main(src_path: str, dst_path: str):
    try:
        reader = GGUFReader(src_path, 'r')
    except Exception as e:
        print(f"read file error: {e}", file=sys.stderr)
        return

    tensors_to_keep: list[ReaderTensor] = [t for t in reader.tensors if not looks_like_moe(t.name)]
    if len(tensors_to_keep) == len(reader.tensors):
        print("    COPY!")

    alignment = reader.alignment
    endian_char = '<' if reader.endianess == GGUFEndian.LITTLE else '>'

    with open(dst_path, 'wb') as fout:
        kv_count = int(reader.fields['GGUF.kv_count'].parts[-1][0])
        gguf_version = int(reader.fields['GGUF.version'].parts[-1][0])

        # GGUF magic has a fixed byte order, even for big-endian files.
        new_header = struct.pack('<I', GGUF_MAGIC) + struct.pack(
            f"{endian_char}I Q Q", gguf_version, len(tensors_to_keep), kv_count
        )
        fout.write(new_header)

        header_size = len(new_header)
        first_tensor_info_offset = reader.tensors[0].field.offset if reader.tensors else reader.data_offset
        metadata_size = first_tensor_info_offset - header_size

        metadata_bytes = reader.data[header_size:header_size + metadata_size]
        fout.write(metadata_bytes)

        current_data_offset = 0
        for tensor in tensors_to_keep:
            name_bytes = tensor.name.encode('utf-8')
            fout.write(struct.pack(f"{endian_char}Q", len(name_bytes)))
            fout.write(name_bytes)
            fout.write(struct.pack(f"{endian_char}I", len(tensor.shape)))
            for dim in (tensor.shape):
                fout.write(struct.pack(f"{endian_char}Q", dim))
            fout.write(struct.pack(f"{endian_char}I", tensor.tensor_type.value))
            fout.write(struct.pack(f"{endian_char}Q", current_data_offset))

            current_data_offset += tensor.n_bytes
            current_data_offset = align_offset(current_data_offset, alignment)

        tensor_info_end_offset = fout.tell()
        data_block_start_offset = align_offset(tensor_info_end_offset, alignment)
        padding_size = data_block_start_offset - tensor_info_end_offset
        if padding_size > 0:
            fout.write(b'\x00' * padding_size)

        with open(src_path, 'rb') as fin:
            total_data_written = 0
            for tensor in tensors_to_keep:
                fin.seek(tensor.data_offset)
                tensor_bytes = fin.read(tensor.n_bytes)
                if len(tensor_bytes) != tensor.n_bytes:
                    raise IOError(f"Error: expect {tensor.n_bytes} bytes, but get {len(tensor_bytes)}")

                fout.write(tensor_bytes)
                total_data_written += len(tensor_bytes)

                padding_to_add = align_offset(tensor.n_bytes, alignment) - tensor.n_bytes
                if padding_to_add > 0:
                    fout.write(b'\x00' * padding_to_add)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("src", help="input path")
    parser.add_argument("dst", help="output path")
    args = parser.parse_args()

    main(args.src, args.dst)
