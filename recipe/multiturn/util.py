import asyncio


async def run_async(f):
    return await asyncio.get_running_loop().run_in_executor(None, f)
