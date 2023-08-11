import asyncio
import time
  

async def main():
    task = asyncio.create_task(other_function())
    return_value = await task    
    print("A")
    await asyncio.sleep(2)
    print("B")    
    print(return_value)
async def other_function():
    await asyncio.sleep(1)
    print("1")
    await asyncio.sleep(2)
    print("2")
    return 10
start = time.time()
asyncio.run(main())
print("total running time: ", time.time() - start)