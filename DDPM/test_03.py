from datetime import datetime
 
# 获取当前时间
now = datetime.now()
 
# 格式化为字符串
formatted_string = now.strftime("%Y-%m-%d %H:%M:%S")
print(formatted_string)  # 输出类似 "2023-04-01 12:34:56"
print(type(formatted_string))

s = 'work_dirs/diffusion_'
print(type(s))

