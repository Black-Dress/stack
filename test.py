from contextlib import redirect_stdout
import os
import baostock as bs
import akshare as ak

def silent_login():
    """静默登录baostock"""
    with open(os.devnull, 'w') as f, redirect_stdout(f):
        lg = bs.login()
    if lg.error_code != '0':
        print(f'登录失败: {lg.error_msg}')
        return False
    return True

def silent_logout():
    """静默登出baostock"""
    with open(os.devnull, 'w') as f, redirect_stdout(f):
        bs.logout()
        
def get_etf_history_ak(code, start_date, end_date):
    """
    使用 akshare 获取 ETF 历史数据
    code 格式：sh512480 或 512480
    """
    # 转换代码格式：akshare 通常使用 'sh512480' 或 'sz159915'
    # 如果您的 code 是 'sh.512480'，需要转换为 'sh512480'
    ak_code = code.replace('.', '')
    df = ak.fund_etf_hist_em(symbol=ak_code, period="daily", 
                              start_date=start_date.replace('-', ''),
                              end_date=end_date.replace('-', ''),
                              adjust="qfq")  # 前复权
    return df

def main():
    silent_login()
    rs = bs.query_history_k_data_plus("sh.512480", "date,close", 
                                   start_date="2025-01-01", 
                                   end_date="2025-12-31")
    if rs.error_code != '0':
        print(f"错误: {rs.error_msg}")
    else:
        data_list = []
        while (rs.error_code == '0') & rs.next():
            data_list.append(rs.get_row_data())
        if data_list:
            print(f"获取到 {len(data_list)} 条记录")
            for row in data_list[:5]:  # 打印前5条
                print(row)
        else:
            print("没有获取到数据")
            
    silent_logout()
if __name__ == '__main__':
    main()