import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
data = pd.read_csv(r"C:\Users\Shlok\Downloads\host_metrics 1.csv")
data_filt = data[~data['host_id'].isin([1, 2])].reset_index(drop=True)
print(data_filt.head(10))
data_filt = data_filt.head(2000)
x=data_filt.ts
y=data_filt.memory_usage_pct
# data = pd.read_csv(r"C:\server_data\DataCenter Last month data\SERVER WISE DATA\MEMORY DATA\superadmin-memory-custom (1).csv")
# data.drop(axis=0, columns=["hostId", "hostName"], inplace=True)
# print(data.head)
# y=data.memoryUsagePct
# x=data.timestamp
plt.plot(x,y)


# data1 = pd.read_csv(r"C:\server_data\DataCenter Last month data\SERVER WISE DATA\MEMORY DATA\superadmin-memory-custom (2).csv")
# data1.drop(axis=0, columns=["hostId", "hostName"], inplace=True)
# y=data1.memoryUsagePct
# x=data1.timestamp
# plt.plot(x,y)


# data2 = pd.read_csv(r"C:\server_data\DataCenter Last month data\SERVER WISE DATA\MEMORY DATA\superadmin-memory-custom.csv")
# data2.drop(axis=0, columns=["hostId", "hostName"], inplace=True)
# y=data2.memoryUsagePct
# x=data2.timestamp
# plt.plot(x,y)

plt.title("Plotting for memory usage")
plt.xlabel("time")
plt.ylabel("memory usage")
plt.show()
print(data.columns)
