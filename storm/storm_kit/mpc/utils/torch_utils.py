#
# MIT License
#
# Copyright (c) 2020-2021 NVIDIA CORPORATION.
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.#
import torch

def find_first_idx(array, value):
    nonzero_indices = torch.nonzero(array > value, as_tuple=False)
    if nonzero_indices.size(0) == 0:
        raise ValueError("No elements found greater than the specified value.")
    f_idx = nonzero_indices[0].item()
    return f_idx

def find_last_idx(array, value):
    print(array, value)
    nonzero_indices = torch.nonzero(array <= value, as_tuple=False)
    if nonzero_indices.size(0) == 0:
        raise ValueError("No elements found less than or equal to the specified value.")
    f_idx = nonzero_indices[-1].item()
    return f_idx
