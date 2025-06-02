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
