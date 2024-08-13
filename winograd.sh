#!/bin/bash

# 检查输入参数数量是否正确
if [ "$#" -ne 8 ]; then
    echo "Error: Invalid number of parameters."
    echo "Usage: $0 <batch size> <input channel> <output channel> <input height> <input width> <tune num> <ARCH(x86/arm/amd)> <thread>"
    exit 1
fi

# 输入参数
b=$1
ic=$2
oc=$3
H=$4
W=$5
n=$6
a=$7
t=$8

# 计算TN_f63值
H_f63_div_6=$(echo "scale=2; $H/6" | bc)
W_f63_div_6=$(echo "scale=2; $W/6" | bc)
TN_f63_H=$(echo "($H_f63_div_6 + 0.999)/1" | bc)  # 向上取整H/6
TN_f63_W=$(echo "($W_f63_div_6 + 0.999)/1" | bc)  # 向上取整W/6
TN_f63=$(echo "$TN_f63_H * $TN_f63_W" | bc)

# 计算TN_f43值
H_f43_div_4=$(echo "scale=2; $H/4" | bc)
W_f43_div_4=$(echo "scale=2; $W/4" | bc)
TN_f43_H=$(echo "($H_f43_div_4 + 0.999)/1" | bc)  # 向上取整H/4
TN_f43_W=$(echo "($W_f43_div_4 + 0.999)/1" | bc)  # 向上取整W/4
TN_f43=$(echo "$TN_f43_H * $TN_f43_W" | bc)

# 使用bc工具计算FLOPs_f63和FLOPs_f43
flops_f63=$(echo "$TN_f63 * $ic * 8 * 8 * 8 * 2 * 2 + $TN_f63 * $ic * $oc * 8 * 8 * 2 + $TN_f63 * $oc * 8 * 6 * (8 + 6) * 2" | bc)
flops_f43=$(echo "$TN_f43 * $ic * 6 * 6 * 6 * 2 * 2 + $TN_f43 * $ic * $oc * 6 * 6 * 2 + $TN_f43 * $oc * 6 * 4 * (6 + 4) * 2" | bc)

# 使用bc工具计算MACs_f63和MACs_f43
macs_f63=$(echo "$ic * $H * $W + 8 * 8 + 2 * 8 * 8 * $TN_f63 * $ic + 8 * 8 * $TN_f63 * $ic + 8 * 8 + 2 * 8 * 8 * $TN_f63 * $ic + 8 * 8 * $ic * $oc + 8 * 8 * $ic * $TN_f63 + 2 * 8 * 8 * $oc * $TN_f63 + 8 * 8 * $oc * $TN_f63 + 8 * 6 + 2 * 8 * 6 * $oc * $TN_f63 + 8 * 6 * $TN_f63 * $oc + 8 * 6 + 2 * $oc * $H * $W" | bc)
macs_f43=$(echo "$ic * $H * $W + 6 * 6 + 2 * 6 * 6 * $TN_f43 * $ic + 6 * 6 * $TN_f43 * $ic + 6 * 6 + 2 * 6 * 6 * $TN_f43 * $ic + 6 * 6 * $ic * $oc + 6 * 6 * $ic * $TN_f43 + 2 * 6 * 6 * $oc * $TN_f43 + 6 * 6 * $oc * $TN_f43 + 6 * 4 + 2 * 6 * 4 * $oc * $TN_f43 + 6 * 4 * $TN_f43 * $oc + 6 * 4 + 2 * $oc * $H * $W" | bc)

# 输出FLOPs和MACs值
echo "Winograd F63 FLOPs: $flops_f63"
echo "Winograd F63 MACs: $macs_f63"
echo "Winograd F43 FLOPs: $flops_f43"
echo "Winograd F43 MACs: $macs_f43"

# 选择脚本逻辑
if (( flops_f63 < flops_f43 )); then
    echo "Executing python3.8 winograd_f63_nhwc_offline_fusion_prepack.py -b $b -ic $ic -oc $oc -H $H -W $W -n $n -a $a -t $t"
    python3 winograd_f63_nhwc_offline_fusion_prepack.py -b $b -ic $ic -oc $oc -H $H -W $W -n $n -a $a -t $t
else
    echo "Executing winograd_f43_nhwc_offline_fusion_prepack.py -b $b -ic $ic -oc $oc -H $H -W $W -n $n -a $a -t $t"
    python3 winograd_f43_nhwc_offline_fusion_prepack.py -b $b -ic $ic -oc $oc -H $H -W $W -n $n -a $a -t $t
fi
