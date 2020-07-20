import pynvml

def getgpuinfo():
  pynvml.nvmlInit()
  handle0 = pynvml.nvmlDeviceGetHandleByIndex(0)
  handle1 = pynvml.nvmlDeviceGetHandleByIndex(1)
  meminfo0 = pynvml.nvmlDeviceGetMemoryInfo(handle0)
  meminfo1 = pynvml.nvmlDeviceGetMemoryInfo(handle1)
  used0 = (meminfo0.used / 1024) /1024
  used1 = (meminfo2.used / 1024) /1024
  result = []
  result.append(used0)
  result.append(used1)
  return result
  
while True:
  result = getgpuinfo()
  print('GPU0 : ' + result[0])
  print('GPU1 : ' + result[1])
  time.sleep(10)
