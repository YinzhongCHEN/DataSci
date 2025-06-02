import torch
def tensor_aggregate(col_tensor: torch.Tensor, func: str):
    """
    对一个张量（1D）执行 func ∈ {"COUNT","SUM","AVG","MIN","MAX"} 并返回标量结果。
    如果 func=="COUNT"，则返回行数；否则对 col_tensor 做 sum/min/max 等。
    """
    # COUNT
    if func == "COUNT":
        return int(col_tensor.shape[0])

    # SUM/MIN/MAX/AVG 需要先把张量搬回 CPU，如果在 GPU 上
    if col_tensor.device.type == "cuda":
        t_cpu = col_tensor.cpu()
    else:
        t_cpu = col_tensor

    if func == "SUM":
        return float(t_cpu.sum().item())
    elif func == "MIN":
        return float(t_cpu.min().item())
    elif func == "MAX":
        return float(t_cpu.max().item())
    elif func == "AVG":
        cnt = t_cpu.shape[0]
        total = float(t_cpu.sum().item())
        return total / cnt if cnt > 0 else 0.0
    else:
        raise ValueError(f"不支持的聚合函数：{func}")