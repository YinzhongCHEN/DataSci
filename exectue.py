from sql_parser import parse_sql_select_with_from, parse_sql_with_window
from tensor_select import collect_where_columns,tensor_select, tensor_window_row_number
from utils import load_data
from sort import sort_and_time
import pandas as pd
import numpy as np
import time
import torch
def execute_sql_with_all_support(sql: str, table_to_path: dict, order_by=None,
                                 descending=False, nrows=None):
    """
    支持三类 SQL：
      1. 窗口函数（ROW_NUMBER OVER ...）   → 张量层面生成 row_num 列并返回
      2. 简单聚合（COUNT/SUM/MIN/MAX/AVG）  → 张量层面做过滤 + 聚合返回
      3. 普通投影（可带 ORDER BY／可带 WHERE）→ 走之前的 SELECT+排序逻辑返回

    参数：
      - sql:           用户传入的 SQL 字符串（只支持以上三种形式，不支持 JOIN/GROUP BY 等）
      - table_to_path: dict[str→str]，表名→CSV 路径
      - order_by:      如果是普通投影且需要排序，此处传排序列列表或字符串
      - descending:    排序是否降序
      - nrows:         可选，只读取前 N 行测试

    返回：
      - result_df:     Pandas DataFrame，包含 SELECT 列 + 可能的窗口列/聚合列
      - result_tensors:张量字典（如果是窗口/聚合查询，需要包含 'row_num' 或聚合结果；如果是普通投影，则含 select_cols 列）
    """
    overall_start = time.time()

    # —— 1. 先尝试窗口函数解析 —— 
    t0 = time.time()
    table_name_w, select_cols_w, where_pred_w, grp_col, sort_col, is_window, alias_w = parse_sql_with_window(sql)
    t1 = time.time()
    # 如果是窗口函数
    if is_window:
        print(f"[TIME] 解析 Window SQL 耗时：{(t1 - t0)*1000.0:.3f} ms")

        # 2. 检查表名
        if table_name_w not in table_to_path:
            raise ValueError(f"未找到表 {table_name_w} 对应的路径，请先在 table_to_path 中定义。")
        csv_path = table_to_path[table_name_w]

        # 3. 构造 usecols = select_cols + [grp_col, sort_col] + WHERE 里用到的列（如果有）
        t2 = time.time()
        where_cols = collect_where_columns(where_pred_w)
        where_cols = collect_where_columns(where_pred_w) or set()
        # 原来的写法会报错：list + set；改为先转换成 list
        usecols = list(set(select_cols_w + [grp_col, sort_col] + list(where_cols)))
        t3 = time.time()
        print(f"[TIME] 提取窗口要加载的列耗时：{(t3 - t2)*1000.0:.3f} ms → usecols={usecols}")

        # 4. 只加载这些列到张量
        t4 = time.time()
        tensors, df_subset = load_data(csv_path, usecols=usecols, nrows=nrows)
        t5 = time.time()
        print(f"[TIME] 加载耗时：{(t5 - t4)*1000.0:.3f} ms → 行数={df_subset.shape[0]}")

        # 5. 在张量上做 WHERE 过滤并投影（先保留 grp_col, sort_col，再传给窗口函数；同时保留 select_cols）
        t6 = time.time()
        filtered = tensor_select(
            tensors=tensors,
            select_cols=select_cols_w + [grp_col, sort_col],
            where_pred=where_pred_w
        )
        t7 = time.time()
        print(f"[TIME] WHERE+投影 耗时：{(t7 - t6)*1000.0:.3f} ms  → 过滤后行数={filtered[grp_col].shape[0]}")

        # 6. 调用 tensor_window_row_number 生成 'row_num'
        t8 = time.time()
        windowed = tensor_window_row_number(
            tensors=filtered,
            grp_col=grp_col,
            sort_col=sort_col
        )
        t9 = time.time()
        print(f"[TIME] ROW_NUMBER 耗时：{(t9 - t8)*1000.0:.3f} ms")

        # 7. 把 windowed 打回 CPU 并封装成 DataFrame 返回
        #    选列：select_cols_w + [grp_col, sort_col, alias_w or 'row_num']
        final_cols = list(select_cols_w) + [grp_col, sort_col, (alias_w or 'row_num')]
        Mf = windowed[grp_col].shape[0]
        data = {}
        for c in final_cols:
            arr = windowed[c]
            if arr.device.type == 'cuda':
                arr = arr.cpu()
            data[c] = arr.numpy()
        result_df = pd.DataFrame(data, index=np.arange(Mf))
        overall_end = time.time()
        print(f"[TIME] 总耗时：{(overall_end-overall_start)*1000.0:.3f} ms")
        print(f"[TIME] 执行耗时：{(overall_end-overall_start-(t5-t4))*1000.0:.3f} ms")
        # return windowed
        return result_df, windowed

    # —— 2. 如果不是窗口函数，再尝试聚合解析 —— 
    t10 = time.time()
    table_name_p, select_cols_p, where_pred_p, agg_func, agg_col, is_agg = parse_sql_select_with_from(sql)
    t11 = time.time()

    if is_agg:
        print(f"[TIME] 解析 Aggregate SQL 耗时：{(t11 - t10)*1000.0:.3f} ms")
        if table_name_p not in table_to_path:
            raise ValueError(f"未找到表 {table_name_p} 对应的路径，请先在 table_to_path 中定义。")
        csv_path = table_to_path[table_name_p]

        # 2. 聚合时需要加载的列
        t12 = time.time()
        where_cols = collect_where_columns(where_pred_p)
        if agg_func == "COUNT":
            # COUNT(*) 只需加载一个非空列就行，我们直接加载 where_cols 中的任意列或 “1 列”
            dummy_col = agg_col or (where_cols.pop() if where_cols else None)
            usecols = [dummy_col] if dummy_col else []
        else:
            # SUM(col)/MIN(col)/etc，需要加载 agg_col + where_cols
            # usecols = list(set([agg_col] + (where_cols or [])))
            where_cols = collect_where_columns(where_pred_p) or set()
            usecols = list(set([agg_col] + list(where_cols)))
        t13 = time.time()
        print(f"[TIME] 提取聚合要加载的列 耗时：{(t13 - t12)*1000.0:.3f} ms → usecols={usecols}")

        # 3. 加载张量
        t14 = time.time()
        if usecols:
            tensors_p, df_subset_p = load_data(csv_path, usecols=usecols, nrows=nrows)
        else:
            # COUNT(*) 且没有 WHERE 引用列，随便加载一列 “l_orderkey” 之类
            tensors_p, df_subset_p = load_data(csv_path, usecols=["l_orderkey"], nrows=nrows)
        t15 = time.time()
        print(f"[TIME] 加载耗时：{(t15 - t14)*1000.0:.3f} ms → 行数={df_subset_p.shape[0]}")

        # 4. 过滤（如果有 WHERE），只保留聚合用到的列
        if where_pred_p is not None:
            t16 = time.time()
            if agg_func == "COUNT":
                dummy = list(tensors_p.keys())[0]
                filtered_p = tensor_select(tensors_p, [dummy], where_pred_p)
                col_tensor = filtered_p[dummy]
            else:
                filtered_p = tensor_select(tensors_p, [agg_col], where_pred_p)
                col_tensor = filtered_p[agg_col]
            t17 = time.time()
            print(f"[TIME] WHERE 过滤 耗时：{(t17 - t16)*1000.0:.3f} ms  → 过滤后行数={col_tensor.shape[0]}")
        else:
            # 直接拿整列
            col_tensor = tensors_p[agg_col] if agg_col else tensors_p[list(tensors_p.keys())[0]]

        # 5. 在张量上做聚合
        t18 = time.time()
        # 简单实现聚合函数
        def tensor_aggregate(col_tensor: torch.Tensor, func: str):
            if func == "COUNT":
                return int(col_tensor.shape[0])
            if col_tensor.device.type == "cuda":
                ct = col_tensor.cpu()
            else:
                ct = col_tensor
            if func == "SUM":
                return float(ct.sum().item())
            if func == "MIN":
                return float(ct.min().item())
            if func == "MAX":
                return float(ct.max().item())
            if func == "AVG":
                cnt = ct.shape[0]
                total = float(ct.sum().item())
                return total / cnt if cnt > 0 else 0.0
            raise ValueError(f"不支持的聚合函数：{func}")

        agg_result = tensor_aggregate(col_tensor, agg_func)
        t19 = time.time()
        print(f"[TIME] 聚合 ({agg_func}) 耗时：{(t19 - t18)*1000.0:.3f} ms")

        # 6. 封装成一行 DataFrame 返回
        result_df = pd.DataFrame({f"{agg_func}": [agg_result]})
        overall_end = time.time()
        print(f"[TIME] 总耗时：{(overall_end-overall_start)*1000.0:.3f} ms")
        print(f"[TIME] 执行耗时：{(overall_end-overall_start-(t15-t14))*1000.0:.3f} ms")
        # return agg_result
        return result_df, None

    # —— 3. 如果既不是窗口函数，也不是聚合，走普通投影/排序分支 —— 
    print(f"[TIME] 非 Window/非 Agg SQL，退回普通解析")
    # a) 先解析 SELECT/WHERE
    if table_name_p is None:
        # 意味着 parse_sql_with_agg 也不认识，抛错
        raise ValueError(f"无法解析的 SQL: {sql}")
    print(f"[TIME] 解析 SELECT/WHERE 耗时：{(t11 - t10)*1000.0:.3f} ms")

    if table_name_p not in table_to_path:
        raise ValueError(f"未找到表 {table_name_p} 对应路径，请先定义。")
    csv_path = table_to_path[table_name_p]

    # b) 提取需要加载的列 = select_cols_p ∪ where_cols
    t20 = time.time()
    where_cols = collect_where_columns(where_pred_p) or set()
    usecols = list(set(select_cols_p + list(where_cols)))
    t21 = time.time()
    print(f"[TIME] 提取常规投影要加载的列耗时：{(t21 - t20)*1000.0:.3f} ms → usecols={usecols}")

    # c) 加载张量
    t22 = time.time()
    tensors_r, df_subset_r = load_data(csv_path, usecols=usecols, nrows=nrows)
    t23 = time.time()
    print(f"[TIME] 加载耗时：{(t23 - t22)*1000.0:.3f} ms → 行数={df_subset_r.shape[0]}")

    # d) 张量 SELECT（WHERE + 投影）
    device0 = next(iter(tensors_r.values())).device
    use_cuda = (device0.type == "cuda" and torch.cuda.is_available())
    if use_cuda:
        start_evt = torch.cuda.Event(enable_timing=True)
        end_evt   = torch.cuda.Event(enable_timing=True)
        start_evt.record()
        result_tensors = tensor_select(tensors_r, select_cols_p, where_pred_p)
        end_evt.record()
        torch.cuda.synchronize()
        sel_ms = start_evt.elapsed_time(end_evt)
        print(f"[TIME] WHERE+投影（GPU） 耗时：{sel_ms:.3f} ms")
    else:
        t24 = time.time()
        result_tensors = tensor_select(tensors_r, select_cols_p, where_pred_p)
        sel_ms = (time.time() - t24)*1000.0
        print(f"[TIME] WHERE+投影（CPU） 耗时：{sel_ms:.3f} ms")

    # e) 如果调用方指定了 order_by，就做排序；否则直接封成 DataFrame 返回
    if not order_by:
        t25 = time.time()
        first_col = select_cols_p[0]
        Mf = result_tensors[first_col].shape[0]
        data = {}
        for c in select_cols_p:
            arr = result_tensors[c]
            if arr.device.type == "cuda":
                arr = arr.cpu()
            data[c] = arr.numpy()
        final_df = pd.DataFrame(data, index=np.arange(Mf))
        t26 = time.time()
        print(f"[TIME] 封装成 DataFrame 耗时：{(t26 - t25)*1000.0:.3f} ms, 返回行数={Mf}")
        overall_end = time.time()
        print(f"[TIME] 总耗时：{(overall_end-overall_start)*1000.0:.3f} ms")
        print(f"[TIME] 执行耗时：{(overall_end-overall_start-(t23-t22))*1000.0:.3f} ms")
        # return result_tensors
        return final_df, result_tensors

    # 如果指定了 order_by，就再做 sort_and_time
    if isinstance(order_by, str):
        order_keys = [order_by]
    else:
        order_keys = order_by

    # 构造一个 DataFrame 供 Pandas 排序对比
    # t27 = time.time()
    first_col = select_cols_p[0]
    Mf = result_tensors[first_col].shape[0]
    df_for_sort = pd.DataFrame({
        c: (result_tensors[c].cpu().numpy() if result_tensors[c].device.type == 'cuda'
            else result_tensors[c].numpy())
        for c in select_cols_p
    }, index=np.arange(Mf))
    # t28 = time.time()
    # print(f"[TIME] 构建排序用 DataFrame 耗时：{(t28 - t27)*1000.0:.3f} ms")

    print(f"\n—— 开始对 SELECT 后结果做排序（order_by={order_keys}）——")
    sorted_df, sorted_tensors = sort_and_time(
        tensors=result_tensors,
        df=df_for_sort,
        sort_key=order_keys,
        descending=descending
    )
    overall_end = time.time()
    print(f"[TIME] 总耗时：{(overall_end-overall_start)*1000.0:.3f} ms")
    print(f"[TIME] 执行耗时：{(overall_end-overall_start-(t23-t22))*1000.0:.3f} ms")
    # return sorted_tensors
    return sorted_df, sorted_tensors
def where_to_query(pred):
    """
    把嵌套的 WHERE 条件元组转换为 Pandas query() 能识别的字符串。
    支持类似：("l_quantity", ">", 50) 以及 ((left),"AND",(right)) 这种嵌套形式。
    """
    if pred is None:
        return ""
    # 单个条件：(col, op, const)
    if isinstance(pred, tuple) and len(pred) == 3 and isinstance(pred[1], str):
        col, op, val = pred
        if isinstance(val, str):
            return f"`{col}` {op} '{val}'"
        else:
            return f"`{col}` {op} {val}"
    # 复合条件：(left_pred, "AND"/"OR", right_pred)
    left, logic, right = pred
    logic = logic.lower()  # "AND"→"and", "OR"→"or"
    return f"({where_to_query(left)}) {logic} ({where_to_query(right)})"


def compare_pandas_tensor(
    sql: str,
    table_to_path: dict,
    order_by=None,
    descending=False,
    nrows=None
):
    """
    独立函数：先用 Pandas 实现 SQL（包含 SELECT/WHERE/(ORDER BY)/窗口/聚合），
    再调用 execute_sql_with_all_support(…) 得到张量版本结果，最后打印两者耗时与结果对比。

    参数：
      - sql:           要测试的 SQL 字符串（仅支持三种形式：窗口、聚合、普通投影+排序）
      - table_to_path: dict[str→str]，表名→CSV 路径
      - order_by:      普通投影时的排序列（字符串或列表），可选
      - descending:    排序是否降序
      - nrows:         限制只读取前 N 行进行测试，可选

    用法示例：
      from compare_utils import compare_pandas_tensor
      compare_pandas_tensor(
          "SELECT l_orderkey, l_linenumber, l_quantity FROM lineitem WHERE l_quantity > 50",
          {"lineitem": r"D:\DataSci\data\lineitem.csv", "orders": r"D:\DataSci\data\orders.csv"},
          order_by=None,
          descending=False,
          nrows=1000
      )
    """
    # —— 1. 先检测是否窗口函数 —— 
    table_name_w, select_cols_w, where_pred_w, grp_col, sort_col, is_window, alias_w = \
        parse_sql_with_window(sql)
    if is_window:
        # 1.1 在 Pandas 端先跑一次：加载、过滤、排序、窗口行号
        usecols = list(set(select_cols_w + [grp_col, sort_col] + list(collect_where_columns(where_pred_w) or set())))
        t0 = time.time()
        df_pd = pd.read_csv(table_to_path[table_name_w], usecols=usecols, nrows=nrows)
        if where_pred_w is not None:
            query_str = where_to_query(where_pred_w)
            df_pd = df_pd.query(query_str)
        df_pd_sorted = df_pd.sort_values([grp_col, sort_col], ascending=[True, False])
        df_pd_sorted["row_num"] = df_pd_sorted.groupby(grp_col).cumcount() + 1
        final_cols = select_cols_w + [grp_col, sort_col, (alias_w or "row_num")]
        df_pd_final = df_pd_sorted[final_cols].reset_index(drop=True)
        t1 = time.time()
        print(f"\n[PANDAS] Window 函数执行耗时：{(t1 - t0)*1000.0:.3f} ms，行数={len(df_pd_final)}")
        print("Pandas 结果前 3 行：")
        print(df_pd_final.head(3))

        # 1.2 调用现有张量版
        print("\n—— 调用张量版本 execute_sql_with_all_support ——")
        t2 = time.time()
        df_tensor, _ = execute_sql_with_all_support(sql, table_to_path, order_by=None, descending=False, nrows=nrows)
        t3 = time.time()
        print(f"[TENSOR] 张量版本执行耗时（仅execute部分）：{(t3 - t2)*1000.0:.3f} ms，行数={len(df_tensor)}")
        print("Tensor 结果前 3 行：")
        print(df_tensor.head(3))

        # 1.3 对比内容是否完全一致
        eq = df_pd_final.reset_index(drop=True).equals(df_tensor.reset_index(drop=True))
        print(f"Window 函数结果一致？ {eq}")
        return

    # —— 2. 否则检测是否聚合 —— 
    table_name_p, select_cols_p, where_pred_p, agg_func, agg_col, is_agg = parse_sql_select_with_from(sql)
    if is_agg:
        # 2.1 Pandas 端：加载、过滤、聚合
        where_cols = collect_where_columns(where_pred_p) or set()
        if agg_func == "COUNT":
            usecols_pd = [agg_col] if agg_col else list(where_cols)[:1] or []
        else:
            usecols_pd = list(set([agg_col] + list(where_cols)))
        t0 = time.time()
        df_pd = pd.read_csv(table_to_path[table_name_p], usecols=usecols_pd, nrows=nrows)
        if where_pred_p is not None:
            query_str = where_to_query(where_pred_p)
            df_pd = df_pd.query(query_str)
        if agg_func == "COUNT":
            pd_val = int(df_pd[agg_col].count()) if agg_col else int(df_pd.shape[0])
        elif agg_func == "SUM":
            pd_val = float(df_pd[agg_col].sum())
        elif agg_func == "MIN":
            pd_val = float(df_pd[agg_col].min())
        elif agg_func == "MAX":
            pd_val = float(df_pd[agg_col].max())
        elif agg_func == "AVG":
            pd_val = float(df_pd[agg_col].mean())
        else:
            raise ValueError(f"不支持的聚合函数：{agg_func}")
        t1 = time.time()
        print(f"\n[PANDAS] 聚合 ({agg_func}) 执行耗时：{(t1 - t0)*1000.0:.3f} ms，结果={pd_val}")

        # 2.2 张量版：调用 execute_sql_with_all_support
        print("\n—— 调用张量版本 execute_sql_with_all_support ——")
        t2 = time.time()
        df_tensor, _ = execute_sql_with_all_support(sql, table_to_path, order_by=None, descending=False, nrows=nrows)
        t3 = time.time()
        tensor_val = df_tensor.iloc[0, 0]
        print(f"[TENSOR] 聚合结果耗时（仅execute部分）：{(t3 - t2)*1000.0:.3f} ms，结果={tensor_val}")

        # 2.3 对比
        if agg_func == "AVG":
            eq = abs(tensor_val - pd_val) < 1e-5
        else:
            eq = (tensor_val == pd_val)
        print(f"聚合 {agg_func} 结果一致？ {eq}")
        return

    # —— 3. 否则走普通投影/排序 —— 
    where_cols = collect_where_columns(where_pred_p) or set()
    usecols_pd = list(set(select_cols_p + list(where_cols)))
    t0 = time.time()
    df_pd = pd.read_csv(table_to_path[table_name_p], usecols=usecols_pd, nrows=nrows)
    if where_pred_p is not None:
        query_str = where_to_query(where_pred_p)
        df_pd = df_pd.query(query_str)
    df_pd_proj = df_pd[select_cols_p]
    if order_by:
        order_list = [order_by] if isinstance(order_by, str) else order_by
        asc_list = [not descending] * len(order_list)
        df_pd_proj = df_pd_proj.sort_values(order_list, ascending=asc_list).reset_index(drop=True)
    else:
        df_pd_proj = df_pd_proj.reset_index(drop=True)
    t1 = time.time()
    print(f"\n[PANDAS] 投影+排序 执行耗时：{(t1 - t0)*1000.0:.3f} ms，行数={len(df_pd_proj)}")
    print("Pandas 结果前 3 行：")
    print(df_pd_proj.head(3))

    # 3.2 Tensor 版：调用 execute_sql_with_all_support
    print("\n—— 调用张量版本 execute_sql_with_all_support ——")
    t2 = time.time()
    df_tensor, _ = execute_sql_with_all_support(sql, table_to_path, order_by=order_by, descending=descending, nrows=nrows)
    t3 = time.time()
    print(f"[TENSOR] 投影+排序 耗时（仅execute部分）：{(t3 - t2)*1000.0:.3f} ms，行数={len(df_tensor)}")
    print("Tensor 结果前 3 行：")
    print(df_tensor.head(3))

    # 3.3 对比
    eq = df_pd_proj.reset_index(drop=True).equals(df_tensor.reset_index(drop=True))
    print(f"Projection/Sort 结果一致？ {eq}")