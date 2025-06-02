import argparse
import pandas as pd
import torch
from sort import sort_and_time
from utils import load_data , all_match
from projection import tensor_projection, pandas_projection
from exectue import execute_sql_on_tensors_with_sort, execute_sql_on_tensors_with_agg
# def main():
#     """
#     主函数：解析命令行参数，调用 sort.py 中的函数完成排序并显示结果。
#     """
#     parser = argparse.ArgumentParser(description="对指定 CSV 文件执行排序，并测量 Pandas 与 Tensor 排序耗时。")
#     parser.add_argument(
#         "--csv", 
#         # required=True, 
#         default= r"D:\DataSci\data\lineitem.csv",
#         help="输入的 lineitem CSV 文件路径（例如：/path/to/lineitem.csv）"
#     )
#     parser.add_argument(
#         "--sort_key", 
#         nargs="+",
#         required=True, 
#         help="排序所使用的列名（例如：l_extendedprice）"
#     )
#     parser.add_argument(
#         "--descending", 
#         action="store_true", 
#         help="是否降序排序，默认是升序。如果指定该参数，则执行降序排序。"
#     )
#     parser.add_argument(
#         "--nrows", 
#         type=int, 
#         default=None, 
#         help="可选：只读取前 N 行进行测试，默认读取全部行。"
#     )
#     args = parser.parse_args()

#     # 加载 DataFrame 并转换为张量,将其导入至GPU上
#     print(f"正在从 {args.csv} 加载数据（前 {args.nrows or '所有'} 行）...")
#     tensors, df = load_data(args.csv,usecols=args.sort_key, nrows=args.nrows)
#     print(f"数据加载完成，共 {df.shape[0]} 行。")

#     # 执行排序并输出耗时
#     # print(f"\n正在对列 {args.sort_key} 进行 {'降序' if args.descending else '升序'} 排序")
#     # sorted_df, sorted_tensors = sort_and_time(
#     #     tensors=tensors, 
#     #     df=df, 
#     #     sort_key=args.sort_key, 
#     #     descending=args.descending
#     # )
#     # sorted_df = sorted_df.reset_index(drop=True)
#     # all_match(sorted_tensors,sorted_df,args.sort_key)
#     proj_cols = ['l_orderkey', 'l_extendedprice']  # 举例

# # 1) Pandas 投影（测时可选）
#     df_proj = pandas_projection(df, proj_cols)
#     print("Pandas 投影后的列：", df_proj.columns.tolist())
#     print("Pandas 投影行数：", len(df_proj))

# # 2) Tensor 投影
#     proj_tensors = tensor_projection(tensors, proj_cols)
#     print("Tensor 投影后的 keys：", list(proj_tensors.keys()))
#     print("Tensor 投影示例（前 3 行）:")
#     for col in proj_cols:
#     # 把前 3 个值搬回 CPU 打印
#       vals = proj_tensors[col][:3].cpu().numpy()
#       print(f"  {col}: {vals}")

# # 3) （可选）验证二者一致性：只要检查每列的前几行即可
#     for col in proj_cols:
#       tensor_vals = proj_tensors[col].cpu().numpy()
#       df_vals     = df_proj[col].to_numpy()
#       if tensor_vals.shape == df_vals.shape and (tensor_vals == df_vals).all():
#         print(f"✅ 列 {col} 的投影结果一致")
#       else:
#         print(f"❌ 列 {col} 的投影结果不一致")

def main():
    parser = argparse.ArgumentParser(description="支持单表聚合（COUNT/SUM/MIN/MAX/AVG）与普通投影查询。")
    parser.add_argument(
        "--sql", 
        required=True,
        help="要执行的 SQL，例如："
             "\"SELECT COUNT(*) FROM lineitem WHERE l_quantity > 25;\" 或\n"
             "\"SELECT SUM(l_extendedprice) FROM lineitem WHERE l_discount < 0.05;\""
    )
    parser.add_argument(
        "--nrows",
        type=int,
        default=None,
        help="可选，只读取前 N 行进行测试"
    )
    args = parser.parse_args()

    # 定义表名 → CSV 路径
    table_to_path = {
        "lineitem": r"D:\DataSci\data\lineitem.csv"
    }

    # 如果用户输入了多条 SQL，用分号拆开
    statements = [st.strip() for st in args.sql.strip().split(";") if st.strip()]
    for stmt in statements:
        print(f"\n—— 执行 SQL：{stmt} ——")
        df_res, _ = execute_sql_on_tensors_with_agg(stmt, table_to_path, nrows=args.nrows)
        print(df_res)
        print("—— 结束 ——\n")

if __name__ == "__main__":
    main()

