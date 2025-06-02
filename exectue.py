from sql_parser import parse_sql_select_with_from
from tensor_select import collect_where_columns,tensor_select
from utils import load_data
from sort import sort_and_time 
from aggregate import tensor_aggregate 
import pandas as pd
import numpy as np
import time
import torch
def execute_sql_on_tensors(sql: str, table_to_path: dict, nrows=None):
    """
    完整流程：
      1) 解析 SQL 得到 table_name, select_cols, where_pred
      2) 根据 select_cols ∪ where_cols 构造 usecols 列表
      3) 调用 load_selected_columns_to_gpu，仅加载 usecols 到张量
      4) 调用 tensor_select 完成过滤 + 投影
      5) 把结果张量搬回 CPU，封装成 Pandas DataFrame 返回

    参数：
      - sql:             用户输入的 SQL 字符串 (简单形式)
      - table_to_path:   dict[str->str]，表名对应 CSV 文件路径，例如 {"lineitem": "D:/data/lineitem.csv"}
      - nrows:           可选，只读前 N 行作为测试

    返回：
      - result_df:       Pandas DataFrame，列为 select_cols，行已过滤
      - result_tensors:  张量字典，只包含 select_cols 且已过滤
    """
    # 1) 解析 SQL
    table_name, select_cols, where_pred = parse_sql_select_with_from(sql)

    # 2) 确认表名存在
    if table_name not in table_to_path:
        raise ValueError(f"未找到表 {table_name} 对应的 CSV 路径，请先在 table_to_path 中定义。")

    # 3) 从 where_pred 中收集所有列名
    where_cols = collect_where_columns(where_pred)
    # 要加载的列 = 投影列 ∪ WHERE 里出现的列
    usecols = list(set(select_cols) | where_cols)

    csv_path = table_to_path[table_name]
    print(f"[INFO] SQL 要用到的列：{usecols}，正在加载 …")

    # 4) 只加载这些列到张量
    t_start_load = time.time()
    tensors, df_subset = load_data(csv_path, usecols=usecols, nrows=nrows)
    t_end_load = time.time()
    load_ms = (t_end_load - t_start_load) * 1000.0
    print(f"[INFO] 共加载 {len(usecols)} 列，耗时：{load_ms:.3f} ms,行数 = {df_subset.shape[0]}")

    # 5) 用 tensor_select 做过滤 + 投影
    device0 = next(iter(tensors.values())).device
    using_cuda = (device0.type == "cuda") and torch.cuda.is_available()
    if using_cuda:
        # GPU 计时
        starter = torch.cuda.Event(enable_timing=True)
        ender   = torch.cuda.Event(enable_timing=True)
        starter.record()
        result_tensors = tensor_select(tensors=tensors, select_cols=select_cols, where_pred=where_pred)
        ender.record()
        torch.cuda.synchronize()
        tensor_ms = starter.elapsed_time(ender)  # 单位：ms
        print(f" SELECT（GPU） 耗时：{tensor_ms:.3f} ms")
    else:
        # CPU 计时
        t_start_tensor = time.time()
        result_tensors = tensor_select(tensors=tensors, select_cols=select_cols, where_pred=where_pred)
        t_end_tensor = time.time()
        tensor_ms = (t_end_tensor - t_start_tensor) * 1000.0
        print(f" SELECT（CPU） 耗时：{tensor_ms:.3f} ms")

    # 6) 把张量结果搬到 CPU，并封装成 DataFrame
    M = result_tensors[select_cols[0]].shape[0]
    data = {}
    for col in select_cols:
        arr = result_tensors[col]
        if arr.device.type == "cuda":
            arr = arr.cpu()
        data[col] = arr.numpy()

    result_df = pd.DataFrame(data, index=np.arange(M))
    return result_df, result_tensors

def execute_sql_on_tensors_with_sort(
    sql: str,
    table_to_path: dict,
    order_by=None,
    descending=False,
    nrows=None
):
    """
    在“SQL→张量过滤+投影”之后，附加一个“排序”步骤，并分别打印各阶段耗时。
    
    参数：
      - sql:           用户输入的 SELECT SQL 字符串（不含 ORDER BY）
      - table_to_path: dict[str->str]，表名对应的 CSV 文件路径
      - order_by:      列名字符串 或 列名列表（在投影列里选择一个或多个列进行排序）
      - descending:    排序是否降序（True 表示降序）
      - nrows:         可选，只读取前 N 行
    
    返回：
      - final_df:      排序后的 Pandas DataFrame
      - final_tensors: 排序后的张量字典
    """
    # —— 1. 解析 SQL —— 
    t_start_parse = time.time()
    table_name, select_cols, where_pred = parse_sql_select_with_from(sql)
    t_end_parse = time.time()
    print(f"[计时] 1. 解析 SQL 耗时：{(t_end_parse - t_start_parse)*1000.0:.3f} ms")

    # —— 2. 提取 where 用到的列，构造 usecols —— 
    t_start_extract = time.time()
    where_cols = collect_where_columns(where_pred)
    usecols = list(set(select_cols) | where_cols)
    t_end_extract = time.time()
    print(f"[计时] 2. 提取列名 耗时：{(t_end_extract - t_start_extract)*1000.0:.3f} ms → usecols={usecols}")

    # —— 3. 检查表名并加载对应列到张量 —— 
    if table_name not in table_to_path:
        raise ValueError(f"无法识别的表名：{table_name}")
    csv_path = table_to_path[table_name]

    t_start_load = time.time()
    tensors, df_subset = load_data(csv_path, usecols=usecols, nrows=nrows)
    t_end_load = time.time()
    print(f"[计时] 3. 加载列到张量耗时：{(t_end_load - t_start_load)*1000.0:.3f} ms → 行数={df_subset.shape[0]}")

    # —— 4. 张量层执行 SELECT（WHERE + 投影） —— 
    device0 = next(iter(tensors.values())).device
    use_cuda = (device0.type == "cuda" and torch.cuda.is_available())

    if use_cuda:
        starter = torch.cuda.Event(enable_timing=True)
        ender   = torch.cuda.Event(enable_timing=True)
        starter.record()
        result_tensors = tensor_select(tensors, select_cols, where_pred)
        ender.record()
        torch.cuda.synchronize()
        select_ms = starter.elapsed_time(ender)
        print(f"[计时] 4. 张量 SELECT（GPU）耗时：{select_ms:.3f} ms")
    else:
        t0 = time.time()
        result_tensors = tensor_select(tensors, select_cols, where_pred)
        select_ms = (time.time() - t0) * 1000.0
        print(f"[计时] 4. 张量 SELECT（CPU）耗时：{select_ms:.3f} ms")

    # 如果没有指定 order_by，就跳过排序，直接封装返回
    if not order_by:
        # 把 result_tensors 打包成 DataFrame
        t_start_pack = time.time()
        first_col = select_cols[0]
        M = result_tensors[first_col].shape[0]
        data = {}
        for col in select_cols:
            arr = result_tensors[col]
            if arr.device.type == 'cuda':
                arr = arr.cpu()
            data[col] = arr.numpy()
        final_df = pd.DataFrame(data, index=np.arange(M))
        t_end_pack = time.time()
        print(f"[计时] 5. 封装成 DataFrame 耗时：{(t_end_pack - t_start_pack)*1000.0:.3f} ms, 返回行数={M}")
        return final_df, result_tensors

    # —— 5. 如果指定了 order_by，就对过滤+投影后的结果做排序 —— 
    #    先把 order_by 规范成列表
    if isinstance(order_by, str):
        order_keys = [order_by]
    else:
        order_keys = order_by

    # 把 result_tensors 转成新的 DataFrame 用于排序对比
    t_start_df_for_sort = time.time()
    first_col = select_cols[0]
    M = result_tensors[first_col].shape[0]
    df_for_sort = pd.DataFrame({
        col: (result_tensors[col].cpu().numpy() if result_tensors[col].device.type=='cuda'
              else result_tensors[col].numpy())
        for col in select_cols
    }, index=np.arange(M))
    t_end_df_for_sort = time.time()
    print(f"[计时] 5. 构建排序用 DataFrame 耗时：{(t_end_df_for_sort - t_start_df_for_sort)*1000.0:.3f} ms")

    # 调用 sort_and_time：它会分别做 Pandas 排序和 Tensor 排序并打印耗时
    print(f"\n—— 开始对 SELECT 后结果做排序（order_by={order_keys}，descending={descending}）——")
    sorted_df, sorted_tensors = sort_and_time(
        tensors=result_tensors,
        df=df_for_sort,
        sort_key=order_keys,
        descending=descending
    )

    # 6. 最后返回排序后的结果
    return sorted_df, sorted_tensors

def execute_sql_on_tensors_with_agg(sql: str, table_to_path: dict, nrows=None):
    """
    支持普通 SELECT（投影+过滤+排序）和单表聚合：
      - 如果 SQL 是 "SELECT COUNT(...)/SUM(...)/MIN(...)/MAX(...)/AVG(...) FROM table [WHERE ...]"，
        则只做 WHERE 过滤然后对指定列(或 *) 聚合并返回结果。
      - 否则认为是普通投影查询，走原先的 SELECT+ORDER BY 流程（如果有 ORDER BY，可改造成 separate）。
    
    这里只示例“一个语句只做聚合”情况，暂不包含 ORDER BY 及排序逻辑（可按需扩展）。
    """
    # 1. 解析 SQL
    t0 = time.time()
    table_name, select_cols, where_pred, agg_func, agg_col, is_agg = parse_sql_select_with_from(sql)
    t1 = time.time()
    print(f"[计时] 1. 解析 SQL 耗时：{(t1 - t0)*1000.0:.3f} ms")

    # 2. 检查表名
    if table_name not in table_to_path:
        raise ValueError(f"未找到表 {table_name} 对应路径，请先在 table_to_path 中定义。")
    csv_path = table_to_path[table_name]

    # 3. 如果是聚合查询，按聚合流程走
    if is_agg:
        # 3.1 提取 WHERE 中用到的列
        t2 = time.time()
        where_cols = collect_where_columns(where_pred)
        # 若聚合是 SUM/MIN/MAX/AVG，需要加载 agg_col；如果是 COUNT(*)，只需把一列用来 COUNT 就行，最简单地用 COUNT(*) 即可
        if agg_func == "COUNT":
            # COUNT(*)：至少加载一个列就能得到行数，比如加载表里任意非空列 l_orderkey
            # 这里我们就加载一个“dummy_col”，比如 select第一列；如果没有列，直接只拿 1 列
            dummy_col = agg_col or (where_cols.pop() if where_cols else None)
            usecols = [dummy_col] if dummy_col else []
        else:
            # SUM/MIN/MAX/AVG(col) 需要加载聚合列本身 + WHERE 可能用到的列
            usecols = list(set([agg_col]) | where_cols)
        t3 = time.time()
        print(f"[计时] 2. 提取聚合要加载的列 耗时：{(t3 - t2)*1000.0:.3f} ms → usecols={usecols}")

        # 3.2 只加载这些列到张量
        t4 = time.time()
        if usecols:
            tensors, df_subset = load_data(csv_path, usecols=usecols, nrows=nrows)
        else:
            # 如果 usecols 为空（意味着 COUNT(*) 且 WHERE 也没引用任何列），
            # 就直接读取整个 CSV 只为计行数：最简单地读取一列行数即可
            # 这里我们假设表至少有一列叫 l_orderkey，把它加载
            tensors, df_subset = load_data(csv_path, usecols=["l_orderkey"], nrows=nrows)
        t5 = time.time()
        print(f"[计时] 3. 加载列到张量耗时：{(t5 - t4)*1000.0:.3f} ms → 行数={df_subset.shape[0]}")

        # 3.3 在张量上做 WHERE 过滤（如果有条件），得到过滤后的张量
        #     为了聚合我们只需要聚合列的张量
        #     若是 COUNT(*)，我们只取任何一个已经加载的列进行行数计数即可
        if where_pred is not None:
            t6 = time.time()
            # 如果是 COUNT(*)：agg_col=None，我们可以在 tensor_select 时把 select_cols 设为一个“dummy”列
            if agg_func == "COUNT":
                dummy = list(tensors.keys())[0]
                filtered = tensor_select(tensors, [dummy], where_pred)
                col_tensor = filtered[dummy]
            else:
                # SUM/MIN/MAX/AVG(col)：投影时只要该列本身就行
                filtered = tensor_select(tensors, [agg_col], where_pred)
                col_tensor = filtered[agg_col]
            t7 = time.time()
            print(f"[计时] 4. 张量 WHERE 过滤耗时：{(t7 - t6)*1000.0:.3f} ms  → 过滤后行数={col_tensor.shape[0]}")
        else:
            # 没有 WHERE：直接对载入整个列做聚合
            col_tensor = tensors[agg_col] if agg_col else tensors[list(tensors.keys())[0]]

        # 3.4 对 col_tensor 做聚合
        t8 = time.time()
        agg_result = tensor_aggregate(col_tensor, agg_func)
        t9 = time.time()
        print(f"[计时] 5. 张量聚合 ({agg_func}) 耗时：{(t9 - t8)*1000.0:.3f} ms")

        # 3.5 把聚合结果封装成只有一行一列的 DataFrame 返回
        result_df = pd.DataFrame({f"{agg_func}": [agg_result]})
        return result_df, None

    # 4. 如果不是聚合查询，说明是普通投影查询（带或不带 WHERE），
    #    可以调用原来那套 “WHERE+投影+排序” 逻辑，或者只执行投影
    else:
        # 4.1 提取 WHERE 用到的列
        t2 = time.time()
        where_cols = collect_where_columns(where_pred)
        usecols = list(set(select_cols) | where_cols)
        t3 = time.time()
        print(f"[计时] 2. 提取投影要加载的列 耗时：{(t3 - t2)*1000.0:.3f} ms → usecols={usecols}")

        # 4.2 加载这些列到张量
        t4 = time.time()
        tensors, df_subset = load_data(csv_path, usecols=usecols, nrows=nrows)
        t5 = time.time()
        print(f"[计时] 3. 加载列到张量耗时：{(t5 - t4)*1000.0:.3f} ms → 行数={df_subset.shape[0]}")

        # 4.3 张量层 SELECT（WHERE + 投影）
        device0 = next(iter(tensors.values())).device
        use_cuda = (device0.type == "cuda" and torch.cuda.is_available())
        if use_cuda:
            starter = torch.cuda.Event(enable_timing=True)
            ender   = torch.cuda.Event(enable_timing=True)
            starter.record()
            result_tensors = tensor_select(tensors, select_cols, where_pred)
            ender.record()
            torch.cuda.synchronize()
            select_ms = starter.elapsed_time(ender)
            print(f"[计时] 4. 张量 SELECT（GPU）耗时：{select_ms:.3f} ms")
        else:
            t6 = time.time()
            result_tensors = tensor_select(tensors, select_cols, where_pred)
            select_ms = (time.time() - t6) * 1000.0
            print(f"[计时] 4. 张量 SELECT（CPU）耗时：{select_ms:.3f} ms")

        # 4.4 最终把 result_tensors 打包成 DataFrame 返回（不做排序）
        t7 = time.time()
        first_col = select_cols[0]
        M = result_tensors[first_col].shape[0]
        data = {}
        for col in select_cols:
            arr = result_tensors[col]
            if arr.device.type == 'cuda':
                arr = arr.cpu()
            data[col] = arr.numpy()
        final_df = pd.DataFrame(data, index=np.arange(M))
        t8 = time.time()
        print(f"[计时] 5. 封装成 DataFrame 耗时：{(t8 - t7)*1000.0:.3f} ms, 返回行数={M}")
        return final_df, result_tensors
table_to_path = {
    "lineitem": r"D:\DataSci\data\lineitem.csv"
}

sql = "SELECT l_orderkey, l_extendedprice FROM lineitem WHERE l_quantity > 25 AND l_discount < 0.05"

# 执行 SQL：会自动解析、只加载用到的列，做过滤 + 投影，最后以 DataFrame 返回
# df_result, tensor_result = execute_sql_on_tensors(sql, table_to_path, nrows=200000)

# print("行数：", df_result.shape[0])
# print("返回的张量字典 keys:", tensor_result.keys())
# df_result, tensor_result = execute_sql_on_tensors_with_sort(sql, table_to_path,order_by='l_extendedprice')