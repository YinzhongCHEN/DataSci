import pandas as pd
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def load_data(csv_path,usecols, nrows=None):
    """
    动态加载 CSV 文件的所有列，并将其转换为 PyTorch 张量、放到 GPU（如果可用）上。
    - 对于整型列，转换为 torch.long。
    - 对于浮点型列，转换为 torch.float。
    - 对于 datetime 列，先转换为 pandas datetime，再转成 int64（纳秒），然后转为 torch.long。
    - 对于字符串/对象型列，转换为 pandas Categorical，再获取 codes (int64)，然后转为 torch.long。
    
    返回：
      - tensors: dict[col_name -> torch.Tensor]，张量均位于 device 上。
      - df: 原始的 pandas DataFrame（保留所有列），方便后续查看原始文本或类别。
    """
    # 1. 先用 pandas 读取 CSV 文件
    df = pd.read_csv(csv_path, usecols=usecols,nrows=nrows)
    
    tensors = {}
    for col in df.columns:
        col_data = df[col]
        dtype = col_data.dtype
        
        # 整型
        if pd.api.types.is_integer_dtype(dtype):
            tensor = torch.from_numpy(col_data.values).to(device)
        
        # 浮点型
        elif pd.api.types.is_float_dtype(dtype):
            tensor = torch.from_numpy(col_data.values).to(device)
        
        # datetime 型 (如果原始不是 datetime，可以尝试解析)
        elif pd.api.types.is_datetime64_any_dtype(dtype):
            # 如果已经是 datetime，直接取 .view('int64') 变为纳秒
            values_int = col_data.view('int64').values
            tensor = torch.from_numpy(values_int).long().to(device)
        
        # 对象/字符串型
        elif pd.api.types.is_object_dtype(dtype):
            # 先把它转成 pandas Categorical，再获取 codes
            cat = col_data.astype('category')
            codes = cat.cat.codes.values.astype('int64')
            tensor = torch.from_numpy(codes).long().to(device)
            # 如果后续需要知道原始字符串含义，可通过 cat.cat.categories 访问
        else:
            # 其他类型(如 bool、category 原本等)，一律先转到 numpy 再通过 torch.tensor
            try:
                tensor = torch.tensor(col_data.values).to(device)
            except Exception as e:
                print(f"无法直接转换列 {col} 的 dtype={dtype}，跳过。错误：{e}")
                continue
        
        tensors[col] = tensor
    # print("目标设备：", device)
    if device.type == 'cuda':
        print("[DEVICE] CUDA 设备名称：", torch.cuda.get_device_name(torch.cuda.current_device()))
    
    return tensors, df