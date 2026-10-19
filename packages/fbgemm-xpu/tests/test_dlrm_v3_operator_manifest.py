# Copyright (c) 2026 Intel Corporation. All Rights Reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Every fbgemm operator of MLPerf DLRM-v3 inference has an XPU kernel.

The list comes from dispatch traces of the generative-recommenders DLRM-v3
inference on XPU (sparse FP16 on CPU with dense replicas, and INT4/INT8
tables sharded row-wise over XPU devices by TorchRec), which call 61
operators on XPU tensors, 8 of them from fbgemm. Losing an XPU registration
for one of them breaks that model, so the check runs on every CI build.
"""

import fbgemm_xpu  # noqa: F401  - registers the fbgemm XPU operators
import pytest
import torch

DLRM_V3_INFERENCE_OPERATORS = [
    "fbgemm::all_to_one_device",
    "fbgemm::asynchronous_complete_cumsum",
    "fbgemm::block_bucketize_sparse_features_inference",
    "fbgemm::bounds_check_indices",
    "fbgemm::dense_to_jagged_forward",
    "fbgemm::int_nbit_split_embedding_codegen_lookup_function",
    "fbgemm::jagged_to_padded_dense_forward",
    "fbgemm::permute_2D_sparse_data",
]


@pytest.mark.parametrize("operator", DLRM_V3_INFERENCE_OPERATORS)
def test_operator_has_xpu_kernel(operator):
    assert torch._C._dispatch_has_kernel_for_dispatch_key(operator, "XPU"), (  # nosec B101
        f"{operator} has no XPU kernel"
    )
