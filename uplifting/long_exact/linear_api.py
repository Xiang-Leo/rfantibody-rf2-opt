"""ctypes interface for the pinned cuBLASLt Linear kernels."""

import ctypes as c


def library(path):
    api = c.CDLL(str(path))
    pointer = c.c_void_p
    api.axial_lt_version.restype = c.c_size_t
    api.axial_lt_count.argtypes = [pointer]
    api.axial_lt_info.argtypes = [pointer, c.c_int, c.POINTER(c.c_uint32), c.POINTER(c.c_uint32),
                                  c.POINTER(c.c_uint32), c.POINTER(c.c_uint64)]
    api.axial_lt_run.argtypes = [pointer, c.c_int, pointer, pointer, pointer, pointer, c.c_uint64, pointer]
    api.axial_lt_destroy.argtypes = [pointer]
    api.axial_lt_destroy.restype = None
    api.linear_lt_create.argtypes = [c.c_int64, c.c_int64, c.c_int64, c.c_int32, pointer,
                                     c.c_int32, c.c_uint64, c.POINTER(c.c_int)]
    api.linear_lt_create.restype = pointer
    api.linear_lt_set_bias.argtypes = [pointer, pointer]
    return api


def info(api, plan, index):
    identifier, tile, stages = c.c_uint32(), c.c_uint32(), c.c_uint32()
    needed = c.c_uint64()
    status = api.axial_lt_info(plan, index, c.byref(identifier), c.byref(tile),
                              c.byref(stages), c.byref(needed))
    return {"index": index, "id": identifier.value, "tile": tile.value,
            "stages": stages.value, "workspace": needed.value, "info_status": status}
