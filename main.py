import argparse
from exectue import execute_sql_with_all_support
def main():
    parser = argparse.ArgumentParser(description="支持窗口函数、聚合、投影、排序的 SQL 处理器")
    parser.add_argument(
        "--sql", required=True,
        help="要执行的 SQL，支持三种形式：\n"
             "  1) 带窗口函数：SELECT ..., ROW_NUMBER() OVER (PARTITION BY grp_col ORDER BY sort_col DESC) AS row_num FROM table [WHERE ...];\n"
             "  2) 聚合：SELECT COUNT(*) / SUM(col) / MIN(col) / MAX(col) / AVG(col) FROM table [WHERE ...];\n"
             "  3) 普通投影：SELECT col1, col2 FROM table [WHERE ...] [ORDER BY col1 [DESC]]"
    )
    parser.add_argument(
        "--order_by", nargs="+",
        help="普通投影时可选的排序列（多列或单列）"
    )
    parser.add_argument(
        "--desc", action="store_true",
        help="如果指定了 --order_by，设置降序排序；否则升序"
    )
    parser.add_argument(
        "--nrows", type=int, default=None,
        help="只读取前 N 行做测试"
    )
    args = parser.parse_args()

    # 表名→CSV 路径映射，根据你项目实际情况修改
    table_to_path = {
        "lineitem": r"D:\DataSci\data\lineitem.csv",
        "orders":    r"D:\DataSci\data\orders.csv"
        # 你可以继续添加其它表
    }

    # df_result, 
    tensor_result = execute_sql_with_all_support(
        sql=args.sql,
        table_to_path=table_to_path,
        order_by=args.order_by,
        descending=args.desc,
        nrows=args.nrows
    )

    # print("\n===== 最终返回 DataFrame =====")
    # print(df_result.head())
    # print(f"(共 {df_result.shape[0]} 行，{df_result.shape[1]} 列)")
    # 如果需要查看 tensor 输出，可以打印 tensor_result.keys()

if __name__ == "__main__":
    main()

