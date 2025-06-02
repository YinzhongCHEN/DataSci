import torch
def tensor_select(tensors: dict, select_cols: list, where_pred):
    """
    在张量层面实现 SELECT select_cols FROM table WHERE where_pred。
    - tensors: {col_name: Tensor}, 形状 [N]
    - select_cols:  要投影的列名列表
    - where_pred:   嵌套元组，见 parse_sql_select_with_from。如果为 None，表示不做过滤。

    返回：
      - projected_tensors: 只包含 select_cols 的张量字典，且已经应用过滤条件
    """
    # 1) 先计算布尔掩码 mask；如果 where_pred 为 None，就全选
    N = next(iter(tensors.values())).shape[0]
    if where_pred is None:
        mask = torch.ones(N, dtype=torch.bool, device=next(iter(tensors.values())).device)
    else:
        def eval_pred(pred):
            # 三种形式：("col", op, const)；(pred1, "AND"/"OR", pred2)；("NOT", pred, None) 或 (pred, "NOT", None)
            if isinstance(pred, tuple) and len(pred) == 3 and pred[1] in (">","<",">=","<=","==","!="):
                col, op, const = pred
                col_vals = tensors[col]
                if op == ">":   return col_vals > const
                if op == "<":   return col_vals < const
                if op == ">=":  return col_vals >= const
                if op == "<=":  return col_vals <= const
                if op == "==":  return col_vals == const
                if op == "!=":  return col_vals != const
            if isinstance(pred, tuple) and len(pred) == 3 and pred[1] in ("AND","OR"):
                left, logic, right = pred
                ml = eval_pred(left)
                mr = eval_pred(right)
                return ml & mr if logic == "AND" else ml | mr
            if isinstance(pred, tuple) and len(pred) == 2 and pred[1] == "NOT":
                sub = pred[0]
                return ~eval_pred(sub)
            raise ValueError(f"无法解析谓词: {pred}")

        mask = eval_pred(where_pred)

    # 2) 把 mask 转成行索引 idx
    idx = torch.nonzero(mask, as_tuple=False).squeeze(1)

    # 3) 对 select_cols 中的每个列做 index_select
    projected_tensors = {
        col: tensors[col].index_select(dim=0, index=idx)
        for col in select_cols
    }
    return projected_tensors

def tensor_select(tensors: dict, select_cols: list, where_pred):
    """
    在张量层面实现 SELECT select_cols FROM table WHERE where_pred。
    - tensors: {col_name: Tensor}, 形状 [N]
    - select_cols:  要投影的列名列表
    - where_pred:   嵌套元组，见 parse_sql_select_with_from。如果为 None，表示不做过滤。

    返回：
      - projected_tensors: 只包含 select_cols 的张量字典，且已经应用过滤条件
    """
    # 1) 先计算布尔掩码 mask；如果 where_pred 为 None，就全选
    N = next(iter(tensors.values())).shape[0]
    if where_pred is None:
        mask = torch.ones(N, dtype=torch.bool, device=next(iter(tensors.values())).device)
    else:
        def eval_pred(pred):
            # 三种形式：("col", op, const)；(pred1, "AND"/"OR", pred2)；("NOT", pred, None) 或 (pred, "NOT", None)
            if isinstance(pred, tuple) and len(pred) == 3 and pred[1] in (">","<",">=","<=","==","!="):
                col, op, const = pred
                col_vals = tensors[col]
                if op == ">":   return col_vals > const
                if op == "<":   return col_vals < const
                if op == ">=":  return col_vals >= const
                if op == "<=":  return col_vals <= const
                if op == "==":  return col_vals == const
                if op == "!=":  return col_vals != const
            if isinstance(pred, tuple) and len(pred) == 3 and pred[1] in ("AND","OR"):
                left, logic, right = pred
                ml = eval_pred(left)
                mr = eval_pred(right)
                return ml & mr if logic == "AND" else ml | mr
            if isinstance(pred, tuple) and len(pred) == 2 and pred[1] == "NOT":
                sub = pred[0]
                return ~eval_pred(sub)
            raise ValueError(f"无法解析谓词: {pred}")

        mask = eval_pred(where_pred)

    # 2) 把 mask 转成行索引 idx
    idx = torch.nonzero(mask, as_tuple=False).squeeze(1)

    # 3) 对 select_cols 中的每个列做 index_select
    projected_tensors = {
        col: tensors[col].index_select(dim=0, index=idx)
        for col in select_cols
    }
    return projected_tensors
def collect_where_columns(where_pred):
    """
    递归遍历 where_pred，把所有出现过的列名收集到一个 set 中。
    并不会解析别名、表名前缀，只要元组里形如 (col, op, const)，就把 col 加进来。
    """
    cols = set()
    if where_pred is None:
        return cols

    # 简单判断：如果是三元组且第二个元素是操作符，就把第一个当成列名
    if isinstance(where_pred, tuple) and len(where_pred) == 3 and where_pred[1] in (">","<",">=","<=","==","!="):
        cols.add(where_pred[0])
    elif isinstance(where_pred, tuple) and len(where_pred) == 3 and where_pred[1] in ("AND","OR"):
        cols |= collect_where_columns(where_pred[0])
        cols |= collect_where_columns(where_pred[2])
    elif isinstance(where_pred, tuple) and len(where_pred) == 2 and where_pred[1] == "NOT":
        cols |= collect_where_columns(where_pred[0])

    return cols
def tensor_window_row_number(
    tensors: dict,
    grp_col: str,
    sort_col: str
):
    """
    对于传入的张量字典 tensors，计算
      ROW_NUMBER() OVER (PARTITION BY grp_col ORDER BY sort_col DESC)
    并把结果放在一个新列 'row_num' 里，返回一个新的张量字典 windowed。

    参数：
      - tensors:  {col_name: torch.Tensor(M,)}，所有张量位于同一 device
      - grp_col:  分组列名（字符串），必须在 tensors 中
      - sort_col: 排序列名（字符串），必须在 tensors 中

    返回：
      - windowed: 在原 tensors 基础上多了一列 'row_num' 的字典
                  其中 windowed['row_num'][i] 即原始第 i 行在其 grp_col 组内
                  按 sort_col 倒序时的排名（从 1 开始）。
    """
    device = tensors[grp_col].device
    M = tensors[grp_col].shape[0]

    # —— Step 1. 先对 sort_col 做一次降序排序，得到索引 idx1 —— 
    sort_vals = tensors[sort_col]  # Shape [M]
    sorted_vals1, idx1 = torch.sort(sort_vals, descending=True)
    # 同时把 grp_col 也应用 idx1
    grp_vals1 = torch.index_select(tensors[grp_col], dim=0, index=idx1)

    # —— Step 2. 再对 grp_vals1 做一次升序稳定排序，得到索引 idx2 —— 
    sorted_grp_vals, idx2 = torch.sort(grp_vals1, descending=False)
    # 合并得到最终索引 idx_final，使得 idx_final[i] = 原始表中排好顺序后第 i 行对应的行号
    idx_intermediate = idx1.index_select(dim=0, index=idx2)
    idx_final = idx_intermediate  # Shape [M]

    # —— Step 3. 用 idx_final 对所有列整体重排 —— 
    sorted_tensors = {
        col: torch.index_select(col_tensor, dim=0, index=idx_final)
        for col, col_tensor in tensors.items()
    }
    sorted_grp = sorted_tensors[grp_col]  # 形状 [M], 已按 grp 升序、组内 sort_col 降序排列

    # —— Step 4. 计算每个 grp 的连续区间长度 counts —— 
    unique_vals, counts = torch.unique_consecutive(sorted_grp, return_counts=True)
    # unique_vals.shape = [G], counts.shape = [G]，G 是分组数量

    # —— Step 5. 为每个分组生成组内行号 1..c 并填进 row_num_sorted —— 
    row_num_sorted = torch.empty(M, dtype=torch.int64, device=device)
    offset = 0
    for c in counts.tolist():
        if c <= 0:
            continue
        rn = torch.arange(1, c + 1, dtype=torch.int64, device=device)
        row_num_sorted[offset : offset + c] = rn
        offset += c

    # —— Step 6. 把排好序的 row_num_sorted 重新映射回原始顺序 —— 
    inv_idx = torch.empty_like(idx_final)
    inv_idx[idx_final] = torch.arange(M, device=device)
    row_num_orig = row_num_sorted.index_select(dim=0, index=inv_idx)

    # —— Step 7. 在原始 tensors 上新增一列 'row_num' —— 
    windowed = dict(tensors)  # 浅拷贝
    windowed['row_num'] = row_num_orig  # 与原始并行的张量，shape = [M]

    return windowed