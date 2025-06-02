import pandas as pd
import torch

def tensor_projection(tensors: dict, proj_cols: list):
    """
    从 tensors（字典）里只取 proj_cols 指定的列，返回一个新的张量字典。

    参数：
      - tensors:   {col_name: Tensor}, 所有张量在 dim=0 上长度相同
      - proj_cols: 要保留的列名列表，必须都是 tensors 中的 key

    返回：
      - projected_tensors: {col_name: Tensor}，只包含 proj_cols 里的列
    """
    # 直接从原始字典里根据 proj_cols 抽出对应张量
    projected_tensors = {col: tensors[col] for col in proj_cols}
    return projected_tensors


def pandas_projection(df: pd.DataFrame, proj_cols: list):
    """
    从 Pandas DataFrame 里只保留 proj_cols 指定的列，并重置索引。
    """
    return df[proj_cols].copy().reset_index(drop=True)
