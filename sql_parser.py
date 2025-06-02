import re

def parse_sql_select_with_from1(sql: str):
    """
    将一个最简单的 SQL SELECT 语句解析成 (table_name, select_cols, where_pred)：
      - table_name:   FROM 后面的表名（只支持一个表，且不考虑别名）
      - select_cols:  投影列列表 ["col1", "col2", ...]
      - where_pred:   一个嵌套元组，表示 WHERE 条件；如果没有 WHERE，则为 None。
    
    支持的 SQL 形式（大小写不敏感）：
      SELECT col1, col2, ... 
      FROM table_name
      [WHERE cond1 [AND|OR cond2 [AND|OR cond3 … ]]]
    
    cond 的形式：col OP const，其中 OP ∈ {>, <, >=, <=, =, !=}。
    多个 cond 可以用 AND 或 OR 连接，不支持括号嵌套。
    
    返回：
      - table_name:  表名字符串（原样，不做大小写转换）
      - select_cols: 列名列表（去除空格）
      - where_pred:  如果有 WHERE，则返回一个递归表达的元组；否则返回 None。
    
    示例：
      sql = "SELECT l_orderkey, l_extendedprice FROM lineitem WHERE l_quantity > 25 AND l_discount < 0.05"
      parse_sql_select_with_from(sql)
      → ("lineitem",
         ["l_orderkey","l_extendedprice"],
         (("l_quantity", ">", 25.0), "AND", ("l_discount", "<", 0.05)) )
    """
    # 1. 去掉首尾空格、末尾分号
    sql = sql.strip().rstrip(";").strip()
    # 用正则拆分 SELECT ... FROM ... WHERE ... (忽略大小写)
    pattern = re.compile(
        r"SELECT\s+(?P<cols>.*?)\s+FROM\s+(?P<table>\S+)"
        r"(?:\s+WHERE\s+(?P<where>.*))?$",
        flags=re.IGNORECASE
    )
    m = pattern.match(sql)
    if not m:
        raise ValueError(f"无法解析的 SQL 语句: {sql}")

    cols_part = m.group("cols").strip()
    table_name = m.group("table").strip()
    where_part = m.group("where")  # 如果没有 WHERE，则为 None

    # 2. 解析 select 列列表
    select_cols = [c.strip() for c in cols_part.split(",") if c.strip()]

    # 3. 解析 WHERE 部分（和之前逻辑相同）
    if where_part is None:
        where_pred = None
    else:
        tokens = re.split(r"\s+", where_part.strip())

        def parse_simple_condition(tok_list, start_idx):
            """
            从 tok_list[start_idx:] 解析一个简单条件 col OP const，
            返回 (cond_tuple, next_index)。
            """
            if start_idx + 2 >= len(tok_list):
                raise ValueError(f"无法解析简单条件： {' '.join(tok_list[start_idx:])}")
            col = tok_list[start_idx]
            op = tok_list[start_idx + 1]
            val_tok = tok_list[start_idx + 2]
            # 处理字符串常量
            if (val_tok.startswith("'") and val_tok.endswith("'")) or \
               (val_tok.startswith('"') and val_tok.endswith('"')):
                const = val_tok[1:-1]
            else:
                # 试着转成 int 或 float
                if re.match(r"^-?\d+$", val_tok):
                    const = int(val_tok)
                else:
                    try:
                        const = float(val_tok)
                    except:
                        const = val_tok
            return (col, op, const), start_idx + 3

        idx = 0
        left_pred, idx = parse_simple_condition(tokens, 0)
        current = left_pred
        while idx < len(tokens):
            logic = tokens[idx].upper()
            if logic not in ("AND", "OR"):
                raise ValueError(f"预期 AND/OR，实际为：{logic}")
            idx += 1
            right_pred, idx = parse_simple_condition(tokens, idx)
            current = (current, logic, right_pred)
        where_pred = current

    return table_name, select_cols, where_pred
# sql = "SELECT l_orderkey, l_extendedprice FROM lineitem WHERE l_quantity > 25 AND l_discount < 0.05"
# table_name, select_cols,where_pred = parse_sql_select_with_from(sql)
# print(table_name)
# print(select_cols)
# print(where_pred)

def parse_sql_select_with_from(sql: str):
    """
    将简单的 SELECT 语句解析为 (table_name, select_cols, where_pred, agg_func, agg_col, is_aggregate)：
      - table_name:   FROM 后面的表名（不支持别名）
      - select_cols:  普通形式时的投影列列表；聚合时可设为 [] 或 None
      - where_pred:   嵌套元组表示的 WHERE 条件；无 WHERE 则为 None
      - agg_func:     如果是聚合语句，取 "COUNT"/"SUM"/"MIN"/"MAX"/"AVG"；否则为 None
      - agg_col:      如果是 SUM/MIN/MAX/AVG，则为对应的列名，比如 "l_extendedprice"；如果是 COUNT(*)，为 None
      - is_aggregate: 布尔，True 表示这是一个聚合语句，False 表示普通的SELECT投影语句

    支持的 SQL：
      1) SELECT col1, col2, ... 
         FROM table_name
         [WHERE cond [AND|OR cond ...]]

      2) SELECT COUNT(*) 
         FROM table_name
         [WHERE cond [AND|OR cond ...]]

      3) SELECT {SUM|MIN|MAX|AVG}(col) 
         FROM table_name
         [WHERE cond [AND|OR cond ...]]

    cond 的形式： col OP const，OP ∈ {>,<,>=,<=,==,!=}，不用在操作符两侧强制加空格
    多个 cond 用 AND / OR 连接，不支持括号嵌套。

    返回：
      (table_name, select_cols, where_pred, agg_func, agg_col, is_aggregate)
    """
    # 1. 清理空格、去末尾分号
    sql = sql.strip().rstrip(";").strip()

    # 2. 给所有比较运算符两侧加空格，确保后面 split 时不会把运算符黏在列名或数字上
    def add_spaces(s: str) -> str:
        # 先处理 >=,<=,!=,==
        s = re.sub(r"(>=|<=|!=|==)", r" \1 ", s)
        # 再处理单字符 >,<,=
        s = re.sub(r"(?<![><!=])([><=])(?![>=!=])", r" \1 ", s)
        return s

    sql_fixed = add_spaces(sql)

    # 3. 匹配 “SELECT … FROM … [WHERE …]”
    #    为了同时捕获聚合和普通投影，用两个分支去尝试匹配
    #    优先判断聚合：COUNT(*) 或 SUM(col) 等
    agg_pattern = re.compile(
        r"SELECT\s+(?P<func>COUNT|SUM|MIN|MAX|AVG)\s*\(\s*(?P<arg>\*\s*|[A-Za-z_][A-Za-z0-9_]*)\s*\)"
        r"\s+FROM\s+(?P<table>\S+)"
        r"(?:\s+WHERE\s+(?P<where>.*))?$",
        flags=re.IGNORECASE
    )
    m_agg = agg_pattern.match(sql_fixed)
    if m_agg:
        func = m_agg.group("func").upper()          # COUNT/SUM/MIN/MAX/AVG
        arg  = m_agg.group("arg").strip()
        table_name = m_agg.group("table").strip()
        where_part = m_agg.group("where")           # 可能为 None

        # 统一把 COUNT(*) 的 arg 视作 agg_col=None
        if func == "COUNT" and arg == "*":
            agg_col = None
        else:
            agg_col = arg   # 例如 "l_extendedprice"

        # 解析 WHERE 子句（如果有的话）
        if where_part is None:
            where_pred = None
        else:
            # 给 where_part 中的运算符两侧加空格
            where_fixed = add_spaces(where_part.strip())
            tokens = re.split(r"\s+", where_fixed)

            def parse_simple(tok_list, i):
                if i + 2 >= len(tok_list):
                    raise ValueError(f"无法解析简单条件：{' '.join(tok_list[i:])}")
                col = tok_list[i]
                op  = tok_list[i + 1]
                val_tok = tok_list[i + 2]
                if (val_tok.startswith("'") and val_tok.endswith("'")) or \
                   (val_tok.startswith('"') and val_tok.endswith('"')):
                    const = val_tok[1:-1]
                else:
                    if re.match(r"^-?\d+$", val_tok):
                        const = int(val_tok)
                    else:
                        try:
                            const = float(val_tok)
                        except:
                            const = val_tok
                return (col, op, const), i + 3

            idx = 0
            left_pred, idx = parse_simple(tokens, 0)
            current = left_pred
            while idx < len(tokens):
                logic = tokens[idx].upper()
                if logic not in ("AND", "OR"):
                    raise ValueError(f"预期 AND/OR，实际：{logic}")
                idx += 1
                right_pred, idx = parse_simple(tokens, idx)
                current = (current, logic, right_pred)
            where_pred = current

        return table_name, [], where_pred, func, agg_col, True

    # 4. 如果不是聚合形式，再匹配普通投影：SELECT col1, col2 … FROM … [WHERE …]
    sel_pattern = re.compile(
        r"SELECT\s+(?P<cols>.*?)\s+FROM\s+(?P<table>\S+)"
        r"(?:\s+WHERE\s+(?P<where>.*))?$",
        flags=re.IGNORECASE
    )
    m_sel = sel_pattern.match(sql_fixed)
    if not m_sel:
        raise ValueError(f"无法解析的 SQL: {sql}")

    cols_part  = m_sel.group("cols").strip()
    table_name = m_sel.group("table").strip()
    where_part = m_sel.group("where")

    select_cols = [c.strip() for c in cols_part.split(",") if c.strip()]

    if where_part is None:
        where_pred = None
    else:
        where_fixed = add_spaces(where_part.strip())
        tokens = re.split(r"\s+", where_fixed)

        def parse_simple(tok_list, i):
            if i + 2 >= len(tok_list):
                raise ValueError(f"无法解析简单条件：{' '.join(tok_list[i:])}")
            col = tok_list[i]
            op  = tok_list[i + 1]
            val_tok = tok_list[i + 2]
            if (val_tok.startswith("'") and val_tok.endswith("'")) or \
               (val_tok.startswith('"') and val_tok.endswith('"')):
                const = val_tok[1:-1]
            else:
                if re.match(r"^-?\d+$", val_tok):
                    const = int(val_tok)
                else:
                    try:
                        const = float(val_tok)
                    except:
                        const = val_tok
            return (col, op, const), i + 3

        idx = 0
        left_pred, idx = parse_simple(tokens, 0)
        current = left_pred
        while idx < len(tokens):
            logic = tokens[idx].upper()
            if logic not in ("AND", "OR"):
                raise ValueError(f"预期 AND/OR，实际：{logic}")
            idx += 1
            right_pred, idx = parse_simple(tokens, idx)
            current = (current, logic, right_pred)
        where_pred = current

    return table_name, select_cols, where_pred, None, None, False
def parse_sql_with_window(sql: str):
    """
    把带窗口函数 ROW_NUMBER() OVER (PARTITION BY grp_col ORDER BY sort_col DESC) 
    的简单 SELECT 语句解析成：
      (table_name, select_cols, where_pred, window_grp_col, window_sort_col, is_window, rest...)
    如果输入 SQL 不包含 ROW_NUMBER() OVER(...)，则返回 (None, None, None, None, None, False)
    其余字段留空，由调用方退回去 parse_sql_with_agg。

    支持格式：
      SELECT col1, col2, ..., ROW_NUMBER() OVER (PARTITION BY grp_col ORDER BY sort_col DESC) AS row_num
      FROM table_name
      [WHERE cond [AND|OR cond ...]];

    返回：
      - table_name:       表名
      - select_cols:      普通列列表，不含窗口函数那一列
      - where_pred:       WHERE 逻辑表达式元组，或 None
      - window_grp_col:   PARTITION BY 后的分组列名
      - window_sort_col:  ORDER BY 后的排序列名
      - is_window:        True/False
      - alias_for_window: 窗口函数的列别名（如'row_num'）；如果没有 AS 片段，则取 None
    """
    sql = sql.strip().rstrip(";").strip()
    # 给比较运算符两侧加空格，方便后续解析 WHERE
    def add_spaces(s: str) -> str:
        s = re.sub(r"(>=|<=|!=|==)", r" \1 ", s)
        s = re.sub(r"(?<![><!=])([><=])(?![>=!=])", r" \1 ", s)
        return s

    # 先拆出 SELECT ... FROM ... [WHERE ...] 这三段
    pattern_main = re.compile(
        r"SELECT\s+(?P<body>.*?)\s+FROM\s+(?P<table>\S+)"
        r"(?:\s+WHERE\s+(?P<where>.*))?$",
        flags=re.IGNORECASE | re.DOTALL
    )
    m = pattern_main.match(sql)
    if not m:
        raise ValueError(f"无法解析的 SQL: {sql}")

    body_part  = m.group("body").strip()
    table_name = m.group("table").strip()
    where_part = m.group("where")  # 可能为 None

    # 尝试在 body_part 里匹配窗口函数的语法
    # window_pattern 会把:
    #   "col1, col2, ROW_NUMBER() OVER (PARTITION BY grp_col ORDER BY sort_col DESC) AS row_num"
    # 拆成三部分：普通列列表、grp_col、sort_col、以及窗口列别名
    window_pattern = re.compile(
        r"^(?P<cols>.*?)\s*,\s*"
        r"ROW_NUMBER\s*\(\s*\)\s*OVER\s*\(\s*PARTITION\s+BY\s+(?P<grp>\w+)\s+"
        r"ORDER\s+BY\s+(?P<sort>\w+)\s+DESC\s*\)\s*(?:AS\s+(?P<alias>\w+))?$",
        flags=re.IGNORECASE | re.DOTALL
    )
    mw = window_pattern.match(body_part)
    if mw:
        select_cols_raw = mw.group("cols").strip()
        select_cols = [c.strip() for c in select_cols_raw.split(",") if c.strip()]
        grp_col = mw.group("grp").strip()
        sort_col = mw.group("sort").strip()
        alias_for_window = mw.group("alias")
        # 解析 WHERE 部分
        if where_part is None:
            where_pred = None
        else:
            where_fixed = add_spaces(where_part.strip())
            tokens = re.split(r"\s+", where_fixed)

            def parse_simple(tok_list, i):
                if i + 2 >= len(tok_list):
                    raise ValueError(f"无法解析条件：{' '.join(tok_list[i:])}")
                col = tok_list[i]
                op  = tok_list[i + 1]
                val = tok_list[i + 2]
                if (val.startswith("'") and val.endswith("'")) or \
                   (val.startswith('"') and val.endswith('"')):
                    const = val[1:-1]
                else:
                    if re.match(r"^-?\d+$", val):
                        const = int(val)
                    else:
                        try:
                            const = float(val)
                        except:
                            const = val
                return (col, op, const), i + 3

            idx = 0
            left_pred, idx = parse_simple(tokens, 0)
            cur = left_pred
            while idx < len(tokens):
                logic = tokens[idx].upper()
                if logic not in ("AND", "OR"):
                    raise ValueError(f"预期 AND/OR，实际：{logic}")
                idx += 1
                right_pred, idx = parse_simple(tokens, idx)
                cur = (cur, logic, right_pred)
            where_pred = cur

        return table_name, select_cols, where_pred, grp_col, sort_col, True, alias_for_window

    # 如果没匹配到窗口函数，返回 is_window=False 让调用方退回给 parse_sql_with_agg
    return None, None, None, None, None, False, None