# AutoFastConv

---
## How to build

**Install required build dependencies:**

* git 
* Python3.7.X or Python3.8.X
* LLVM15.0.3 or higher
* TVM (v0.12 Release or higher) 

Please follow the tutorial https://tvm.apache.org/docs/install/from_source.html to install TVM.

**Git clone AutoFastConv repo**

```bash
git clone https://github.com/MrJungle1/AutoFastConv.git
cd AutoFastConv
```

**Test experiment**

It is recommended to set the tune num to 30000, which will achieve better performance.
```bash
Alg-2:
python3 winograd_f63_nhwc.py -b {batch size} -ic {input channel} -oc {output channel} -H {input height} -W {input width} -n {tune num} -a {ARCH(x86/arm/amd)} -t {thread}
example:
python3 winograd_f63_nhwc.py -b 1 -ic 256 -oc 256 -H 56 -W 56 -n 30000 -a x86 -t 1

Alg-3: 
python3 winograd_f63_nhwc_offline_fusion_prepack.py -b {batch size} -ic {input channel} -oc {output channel} -H {input height} -W {input width} -n {tune num} -a {ARCH(x86/arm/amd)} -t {thread}
example:
python3 winograd_f63_nhwc_offline_fusion_prepack.py -b 1 -ic 256 -oc 256 -H 56 -W 56 -n 30000 -a x86 -t 1

Alg-4: 
bash winograd.sh h {batch size} {input channel} {output channel} {input height} {input width} {tune num} {ARCH(x86/arm/amd)}  {thread}
example:
bash winograd.sh 1 256 256 56 56 30000 x86 1
```

**Reproduce performance**

Each time you run a Python script completely, a json file will be generated in the log folder in the current directory. You can use this json file to reproduce the best performance you searched for before. Just add the -l option to specify the corresponding json when running it again.
```bash
example:
python3 winograd_f63_nhwc.py -b 1 -ic 256 -oc 256 -H 56 -W 56 -n 30000 -a x86 -t 1 -l winograd_ansor_N1_H256_W256_IC56_OC56_2024-01-01_00:00:00.log
```

**Test whole framework**

```bash
bash winograd.sh h {batch size} {input channel} {output channel} {input height} {input width} {tune num} {ARCH(x86/arm/amd)}  {thread}
example:
bash winograd.sh 1 256 256 56 56 30000 x86 1
```