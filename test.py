import akshare as ak
import requests
import json

def get_index_akshare(code="sh000001"):
    """获取指数实时行情"""
    try:
        if code == "sh000001":
            df = ak.stock_zh_index_daily(symbol="sh000001")
        elif code == "sh000300":
            df = ak.stock_zh_index_daily(symbol="sh000300")
        if not df.empty:
            return df.iloc[-1]["close"]
        return None
    except Exception as e:
        print(f"AkShare请求失败: {e}")
        return None

def get_realtime_index_tencent(code='sh000001'):
    """
    腾讯财经实时指数
    code格式：sh000001 上证指数，sz399001 深证成指
    """
    try:
        url = f'https://web.ifzq.gtimg.cn/appstock/app/fqkline/get'
        params = {
            'param': f'{code},day,,,1',
            '_var': 'kline_day'
        }
        r = requests.get(url, params=params, timeout=5)
        if r.status_code == 200:
            data = r.text
            # 腾讯返回的是JSONP格式，需要解析
            json_str = data.split('=')[1].strip(';')
            json_data = json.loads(json_str)
            if json_data['code'] == 0:
                return float(json_data['data'][code]['day'][0][2])  # 最新价
        return None
    except Exception as e:
        print(f"腾讯财经接口请求失败: {e}")
        return None

def get_realtime_price_sina_fallback(code):
    """备用新浪接口"""
    try:
        # 尝试不同的新浪接口域名
        domains = ['hq.sinajs.cn', 'hq2.sinajs.cn', 'hq3.sinajs.cn']
        for domain in domains:
            sina_code = code.replace('.', '')
            url = f'http://{domain}/list={sina_code}'
            headers = {'Referer': 'http://finance.sina.com.cn'}
            r = requests.get(url, headers=headers, timeout=3)
            if r.status_code == 200:
                data = r.text
                parts = data.split('"')[1].split(',')
                if len(parts) > 3 and parts[3]:
                    return float(parts[3])
        return None
    except:
        return None

if __name__ == "__main__":
    print(get_realtime_price_sina_fallback('sh000001'))