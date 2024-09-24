import cv2

def main():
    try:
        count = cv2.cuda.getCudaEnabledDeviceCount()
        for i in range(count):
            device_info = cv2.cuda.getDevice(i)
            print(f"CUDA 设备 {i}:")
            print(f"  名称: {device_info.name()}")
            print(f"  计算能力: {device_info.computeCapability()}")
            print(f"  多处理器数量: {device_info.multiProcessorCount()}")
            print(f"  总内存: {device_info.totalMemory() / (1024**3):.2f} GB")
    except cv2.error:
        print("无法获取 CUDA 设备信息")

if __name__ == "__main__":
    main()
