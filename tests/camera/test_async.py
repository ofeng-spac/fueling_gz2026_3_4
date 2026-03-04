import asyncio

class DataHandler:
    # 同步函数：仅接收必要参数 url
    def fetch_data(self, url):
        print(f"同步请求：{url}")
        return f"同步数据：{url} 结果"
    
    # 异步函数：多一个默认参数 is_async（或其他差异化参数）
    async def fetch_data(self, url, is_async=True):
        print(f"异步请求：{url}")
        await asyncio.sleep(1)
        return f"异步数据：{url} 结果"

# 调用方式
handler = DataHandler()
# 调用同步函数（仅传url，匹配同步函数的参数签名）
sync_result = handler.fetch_data("https://example.com")
# 调用异步函数（传url+默认参数，或显式传is_async）
async def main():
    async_result = await handler.fetch_data("https://example.com", is_async=True)  # 匹配异步函数
    print(sync_result, async_result)

asyncio.run(main())