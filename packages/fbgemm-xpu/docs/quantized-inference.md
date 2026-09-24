# Quantized INT4/INT8 inference on XPU

## Overview

`fbgemm_xpu` provides an XPU implementation of
`fbgemm::int_nbit_split_embedding_codegen_lookup_function`.

The operator reads packed INT4 or INT8 embedding rows from XPU memory and
dequantizes them into FP32, FP16, or BF16 output. Quantization remains an
offline CPU step performed with FBGEMM.

The implementation is used by FBGEMM
`IntNBitTableBatchedEmbeddingBagsCodegen` and by TorchRec quantized embedding
collections when their tables and inputs are placed on XPU.

INT4 and INT8 kernels are generated from a shared Jinja2 SYCL template during
the package build.

## Supported configuration

The XPU lookup supports:

- eager execution;
- unweighted, no-bag lookup (`PoolingMode.NONE`);
- INT4 and INT8 packed weights;
- one or more DEVICE-resident tables;
- a uniform embedding dimension across all tables;
- embedding dimensions that are positive multiples of four;
- int32 or int64 indices and offsets;
- FP32, FP16, or BF16 output;
- row alignment that is a power of two from 1 through 128 (default 16).

The dimension boundary matches the documented FBGEMM IntNBit frontend
contract. It is a supported-subset decision rather than a requirement of the
scalar XPU loads or row padding.

## Packed row layout

Each packed row begins with four bytes containing little-endian FP16 scale and
bias values, followed by unsigned quantized values:

```text
+--------+--------+-------------------------+---------+
| scale  | bias   | packed quantized values | padding |
| 2 B    | 2 B    | D * bits / 8 bytes      |         |
+--------+--------+-------------------------+---------+
```

INT4 stores the earlier element in the low nibble. The row stride is:

```text
round_up(4 + D * bits / 8, row_alignment)
```

Table offsets are byte offsets into the flat packed-weight tensor.

## Validation and error behavior

Checked frontend execution uses two complementary operators:

- `fbgemm::bounds_check_indices` validates logical table row counts and applies
  the selected correction or error policy;
- `fbgemm::int_nbit_split_embedding_codegen_lookup_function` validates the
  supported layout and guards physical packed-storage access.

The lookup wrapper rejects unsupported pooling, weighted lookup, nonempty
cache or UVM tensors, non-DEVICE placement, mixed dimensions, unsupported
storage/output types, malformed metadata and offsets, and negative or pruned
indices.

Import the plugin before using frontends that need its XPU registrations:

```python
import torch
import fbgemm_xpu
```

## Build and test

Build and install the package from the repository root:

```bash
uv pip install -e "packages/fbgemm-xpu[test]" \
  --index https://download.pytorch.org/whl/xpu
```

Run the direct and FBGEMM frontend tests on a machine with an available XPU:

```bash
pytest -rsf packages/fbgemm-xpu/tests/test_int_nbit_lookup.py
```

The TorchRec collection integration test requires a compatible TorchRec
installation. It is skipped automatically when TorchRec is unavailable and can
be selected explicitly after installing it:

```bash
pytest -rsf packages/fbgemm-xpu/tests/test_int_nbit_lookup.py \
  -k high_level_quant_embedding_collection
```

## Known limitations

The implementation does not support:

- pooled (`SUM` or `MEAN`) or weighted lookup;
- cache/UVM-backed tables;
- pruning or index remapping;
- mixed embedding dimensions;
- INT2, FP8, FP16, or FP32 weight storage;
- `torch.compile` or FakeTensor execution;
- distributed/multi-device execution.
