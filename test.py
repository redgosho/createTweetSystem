# sysモジュールをインポート
import sys
 
# sys.maxsizeは通常、32ビットでは 2**31 - 1、64ビットでは 2**63 - 1 になります
is64Bit = sys.maxsize > 2 ** 32
if is64Bit:
    print("64bit")
else:
    print("32bit")