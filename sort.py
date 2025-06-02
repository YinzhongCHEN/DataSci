import pandas as pd
import torch
import time

def tensor_sort(tensors: dict, sort_key: str, descending: bool=False):
    """
    对给定的 tensors（字典）按照 sort_key 这一列做排序，然后把所有列都重排，
    返回 (sorted_tensors_dict, sorted_indices_tensor)。

    参数：
      - tensors：{col_name: torch.Tensor}，所有 Tensor 的 shape 在 dim=0 上应该相同。
      - sort_key：要排序的列名（必须是 tensors 中的一个 key）。
      - descending：是否降序排序，False 表示升序。

    返回：
      - sorted_tensors：{col_name: torch.Tensor}，张量已根据 sort_key 的顺序重排。
      - sorted_idx：torch.Tensor，形状=(N,) 的 LongTensor，表示原始行在排序后对应的新索引。
    """
    # 1) 先取出用来做排序的那个“Key”张量
    key_tensor = tensors[sort_key]
    # 2) 直接调用 torch.sort 得到 (sorted_values, sorted_indices)
    #    sorted_indices 形状为 (N,) 的 LongTensor，值在 [0, N-1] 间，表示每个原始行在排序后对应新位置
    sorted_values, sorted_idx = torch.sort(key_tensor, descending=descending)

    # 3) 用 sorted_idx 对所有列做 index_select，将整张表按同样顺序重排
    sorted_tensors = {}
    for col_name, col_tensor in tensors.items():
        # 注意：dim=0 是行的维度
        sorted_tensors[col_name] = torch.index_select(col_tensor, dim=0, index=sorted_idx)

    return sorted_tensors, sorted_idx
def pandas_sort(df, sort_cols):
    return df.sort_values(by=sort_cols, ascending=True).reset_index(drop=True)
def tensor_multi_sort(tensors: dict, sort_keys: list, descending: bool=False):
    """
    对 tensors 做多列联合排序：先按 sort_keys[0]，再按 sort_keys[1]……。
    基本思路：先把多个 key 列“编码”成一个复合键（见下），然后按复合键排序。

    参数：
      - tensors:    {col_name: torch.Tensor}，所有张量第 0 维长度相同。
      - sort_keys:  列名列表，如 ['l_orderkey', 'l_extendedprice']。
      - descending: 是否降序排序，False 表示升序。

    返回：
      - sorted_tensors: 排序后的张量字典。
      - sorted_idx:     排序使用的索引 Tensor。
    """

# 深拷贝一份张量字典，避免原地覆盖
    sorted_tensors = {k: v for k, v in tensors.items()}

    # 先从最次要 key 开始，到最主要 key 逐次排序
    # 例如 sort_keys = ['A','B','C']，这里就会按 C → B → A 的顺序做三次 torch.sort
    for key in reversed(sort_keys):
        # 取当前表中该列张量
        col_tensor = sorted_tensors[key]
        # torch.sort 默认 stable，从 PyTorch 1.6 开始保证“相等元素排序时保持原先相对顺序”
        sorted_vals, idx = torch.sort(col_tensor, descending=descending)
        # 用同一个 idx 把整张表的所有列都重排
        sorted_tensors = {
            cname: torch.index_select(ctensor, dim=0, index=idx)
            for cname, ctensor in sorted_tensors.items()
        }
        final_sorted_idx = idx
    return sorted_tensors, final_sorted_idx
def sort_and_time(
    tensors: dict,
    df: pd.DataFrame,
    sort_key: str,
    descending: bool = False
):
    """
    同时对 Pandas DataFrame 和对应的张量字典进行排序，并且打印各自耗时（ms）。

    参数：
      - tensors:      {col_name: torch.Tensor}，所有张量在 dim=0 上的长度与 df.shape[0] 相同。
      - df:           Pandas DataFrame，列名应包含 sort_key，并且行顺序与 tensors 中的张量对应。
      - sort_key:     用于排序的列名（字符串），既要在 df 中，也要在 tensors 中存在。
      - descending:   是否降序排序，False 表示升序。

    返回：
      - sorted_df_pd: Pandas 排序后并且重置索引的 DataFrame；
      - sorted_tensors: 排序后对应的新张量字典。
    """
    # 1. Pandas 排序并测时
    if isinstance(sort_key, str):
        pd_sort_cols = [sort_key]
    else:
        pd_sort_cols = sort_key
    # start_time = time.time()
    # # 注意：这里只按单列 sort_key 排序，并重置索引
    sorted_df_pd = pandas_sort(df, pd_sort_cols)[df.columns]
    # pandas_elapsed_ms = (time.time() - start_time) * 1000.0

    # 根据是否可用 GPU 选择TIME方式
    use_cuda = torch.cuda.is_available()
    first_tensor = next(iter(tensors.values()))
    tensor_on_cuda = first_tensor.is_cuda
    if use_cuda and tensor_on_cuda:
        # print("GPUTIME")
        starter = torch.cuda.Event(enable_timing=True)
        ender   = torch.cuda.Event(enable_timing=True)
        starter.record()
        # 选择单列还是多列排序
        if len(pd_sort_cols) > 1:
            sorted_tensors, sorted_idx = tensor_multi_sort(
                tensors, sort_keys=pd_sort_cols, descending=descending
            )
        else:
            key = pd_sort_cols[0]
            sorted_tensors, sorted_idx = tensor_sort(
                tensors, sort_key=key, descending=descending
            )
        ender.record()
        torch.cuda.synchronize()
        tensor_elapsed = starter.elapsed_time(ender)
    else:
        # 用 time.time() 走 CPU 分支
        start_cpu = time.time()
        if len(pd_sort_cols) > 1:
            sorted_tensors, sorted_idx = tensor_multi_sort(
                tensors, sort_keys=pd_sort_cols, descending=descending
            )
        else:
            key = pd_sort_cols[0]
            sorted_tensors, sorted_idx = tensor_sort(
                tensors, sort_key=key, descending=descending
            )
        tensor_elapsed = (time.time() - start_cpu) * 1000.0

    # 3. 打印结果
    # print(f"Pandas 排序耗时：{pandas_elapsed_ms:.3f} ms，排序后行数：{sorted_df_pd.shape[0]}")
    print(f"[TIME] Tensor 排序耗时：{tensor_elapsed:.3f} ms，排序后行数：{sorted_tensors[next(iter(sorted_tensors))].shape[0]}")

    return sorted_df_pd, sorted_tensors
    # return sorted_tensors
