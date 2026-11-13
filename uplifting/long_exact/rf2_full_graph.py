"""Reuse the RF2 full-graph topology when the stock top-k must exclude the diagonal."""

from functools import wraps


def install(torch):
    import dgl
    from rfantibody.rf2.network import Track_module

    original = Track_module.make_topk_graph
    cache = {}
    counts = {"full_graph": 0, "stock_graph": 0}

    @wraps(original)
    def make_graph(xyz, pair, idx, top_k=128, kmin=32, eps=1e-6):
        batch, length = pair.shape[:2]
        full = (
            length > 1 and (top_k <= 0 or top_k >= length - 1)
            and kmin <= 900 and eps == 1e-6
            and xyz.is_cuda and xyz.dtype == torch.float32
            and xyz.device == pair.device == idx.device
            and not torch.is_grad_enabled()
        )
        if full:
            # Coordinates in [-200, 200] have off-diagonal distance below 693 Å.
            # The guarded residue-index term adds less than 2 Å, leaving the
            # diagonal (offset 999.9 Å) strictly outside the stock top-k.
            full = bool((xyz.abs().amax() <= 200) & (idx.abs().amax() <= 1_000_000))
        if not full:
            counts["stock_graph"] += 1
            return original(xyz, pair, idx, top_k=top_k, kmin=kmin, eps=eps)

        key = (batch, length, xyz.device)
        if key not in cache:
            residue = torch.arange(length, device=xyz.device)
            row, column = torch.where(residue[:, None] != residue[None, :])
            rows = row.repeat(batch)
            columns = column.repeat(batch)
            blocks = torch.arange(batch, device=xyz.device).repeat_interleave(row.numel())
            offsets = blocks * length
            graph = dgl.graph((offsets + rows, offsets + columns), num_nodes=batch * length)
            cache.clear()
            cache[key] = (graph, blocks, rows, columns)
        graph, blocks, rows, columns = cache[key]
        graph = graph.local_var()
        if batch == 1:
            graph._uplifting_full_graph_length = length
        graph.edata["rel_pos"] = (xyz[blocks, columns, :] - xyz[blocks, rows, :]).detach()
        counts["full_graph"] += 1
        return graph, pair[blocks, rows, columns][..., None]

    Track_module.make_topk_graph = make_graph
    return counts
