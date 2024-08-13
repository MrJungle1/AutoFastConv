import os
import argparse
import time
import numpy as np
import tvm
import tvm.testing
from scipy import signal
from tvm import auto_scheduler, te

target = "llvm -mcpu=skylake-avx512"
dtype = "float32"
dev = tvm.device(target, 0)

N = 1
Input_Channel = 256
Output_Channel = 256
H = 56
W = 56
Num_Tune = 12800

block_size = 6
kernel_size = 3
tile_size = block_size + kernel_size - 1
H_pad = 56
W_pad = 56
nH = (H_pad - kernel_size + 1) // block_size
nW = (W_pad - kernel_size + 1) // block_size
Tile = N * nH * nW


def calc_kernel(k):

    Output_Channel, Input_Channel, _, _ = k.shape
    '''G = np.array(
        [
            [1, 0, 0],
            [-2 / 9, -2 / 9, -2 / 9],
            [-2 / 9, 2 / 9, -2 / 9],
            [1 / 90, 1 / 45, 2 / 45],
            [1 / 90, -1 / 45, 2 / 45],
            [32 / 45, 16 / 45, 8 / 45],
            [32 / 45, -16 / 45, 8 / 45],
            [0, 0, 1],
        ]
    ).astype(dtype)'''

    G = np.array(
        [
            [1, 0, 0],
            [-2 / 9, -2 / 9, -2 / 9],
            [-2 / 9, 2 / 9, -2 / 9],
            [1 / 90, 1 / 45, 2 / 45],
            [1 / 90, -1 / 45, 2 / 45],
            [1 / 45, 1 / 90, 1 / 180],
            [1 / 45, -1 / 90, 1 / 180],
            [0, 0, 1],
        ]
    ).astype(dtype)

    nk = np.zeros((Output_Channel, Input_Channel,
                   tile_size, tile_size), dtype=dtype)
    ret = np.zeros((tile_size, tile_size, Input_Channel,
                    Output_Channel), dtype=dtype)
    for i in range(Output_Channel):
        for j in range(Input_Channel):
            tmp = k[i][j]
            tmp = np.dot(G, tmp)
            tmp = np.dot(tmp, G.T)
            nk[i][j] = tmp
    for i in range(Input_Channel):
        for o in range(Output_Channel):
            for h in range(tile_size):
                for w in range(tile_size):
                    ret[h][w][i][o] = nk[o][i][h][w]
    return ret


def const_matrix(matrix, name="const_matrix"):

    row, col = matrix.shape
    dtype = str(matrix.dtype)
    idxm = tvm.tir.indexmod

    def select_array(i, j):
        now = tvm.tir.const(0.0, dtype)
        for ii in range(row):
            for jj in range(col):
                now = tvm.tir.Select(
                    tvm.tir.all(idxm(i, row) == ii, idxm(j, col) == jj),
                    tvm.tir.const(matrix[ii][jj], dtype),
                    now,
                )
        return now

    return te.compute(
        matrix.shape, select_array, name=name, attrs={"const_matrix": True}
    )


def gen_matrices():
    '''A = np.array(
        [
            [1, 0, 0, 0, 0, 0],
            [1, 1, 1, 1, 1, 1],
            [1, -1, 1, -1, 1, -1],
            [1, 2, 4, 8, 16, 32],
            [1, -2, 4, -8, 16, -32],
            [1, 1/2, 1/4, 1/8, 1/16, 1/32],
            [1, -1/2, 1/4, -1/8, 1/16, -1/32],
            [0, 0, 0, 0, 0, 1],
        ]
    ).astype(dtype)'''

    A = np.array(
        [
            [1, 0, 0, 0, 0, 0],
            [1, 1, 1, 1, 1, 1],
            [1, -1, 1, -1, 1, -1],
            [1, 2, 4, 8, 16, 32],
            [1, -2, 4, -8, 16, -32],
            [32, 16, 8, 4, 2, 1],
            [32, -16, 8, -4, 2, -1],
            [0, 0, 0, 0, 0, 1],
        ]
    ).astype(dtype)

    B = np.array(
        [
            [1, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, -1, 1 / 2, -1 / 2, 2, -2, -1],
            [-21 / 4, 1, 1, 1 / 4, 1 / 4, 4, 4, 0],
            [0, -17 / 4, 17 / 4, -5 / 2, 5 / 2, -5 / 2, 5 / 2, 21 / 4],
            [21 / 4, -17 / 4, -17 / 4, -5 / 4, -5 / 4, -5, -5, 0],
            [0, 1, -1, 2, -2, 1 / 2, -1 / 2, -21 / 4],
            [-1, 1, 1, 1, 1, 1, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 1],
        ]
    ).astype(dtype)

    return (
        const_matrix(
            A,
            "A",
        ),
        const_matrix(
            B,
            "B",
        ),
    )


@auto_scheduler.register_workload
def winograd():
    data = te.placeholder((N, H, W, Input_Channel), name="data")
    U = te.placeholder(
        (tile_size, tile_size, Input_Channel, Output_Channel), name="U")
    A, B = gen_matrices()
    data_pad = tvm.topi.nn.pad(
        data, (0, 1, 1, 0), (0, H_pad - H - 1, W_pad - W - 1, 0), name="data_pad")
    ik1 = te.reduce_axis((0, tile_size), "ik1")
    ik2 = te.reduce_axis((0, tile_size), "ik2")
    V_tmp = te.compute(
        (tile_size, tile_size, Tile, Input_Channel),
        lambda h, w, n, k: te.sum(
            data_pad[n // (nH * nW), n // nW % nH * block_size + ik1, n %
                     nW * block_size + w, k] * B[ik1, h],
            axis=ik1
        ),
        name="V_tmp",
    )

    V = te.compute(
        (tile_size, tile_size, Tile, Input_Channel),
        lambda h, w, n, k: te.sum(V_tmp[h, ik2, n, k] * B[ik2, w], axis=ik2),
        name="V",
    )

    k = te.reduce_axis((0, Input_Channel), "k")

    M = te.compute(
        (tile_size, tile_size, Tile, Output_Channel),
        lambda h, w, n, m: te.sum(U[h, w, k, m] * V[h, w, n, k], axis=k),
        name="M",
        attrs={"layout_free_placeholders": [U]},
    )

    ok1 = te.reduce_axis((0, tile_size), "ok1")
    ok2 = te.reduce_axis((0, tile_size), "ok2")
    OT_tmp = te.compute(
        (block_size, tile_size, Tile, Output_Channel),
        lambda h, w, n, m: te.sum(M[ok1, w, n, m] * A[ok1, h], axis=ok1),
        name="OT_tmp",
    )
    O = te.compute(
        (N, H, W, Output_Channel),
        lambda n, h, w, m: te.sum(
            OT_tmp[h % block_size, ok2, n * nH * nW + h // block_size *
                   nW + w // block_size, m] * A[ok2, w % block_size],
            axis=ok2
        ),
        name="O",
    )
    return [data, U, O]


def gen_func():

    task = auto_scheduler.SearchTask(
        func=winograd, args=(), target=tvm.target.Target(target),
        layout_rewrite_option=auto_scheduler.LayoutRewriteOption.REWRITE_FOR_PRE_TRANSFORMED
    )
    log_dir = "./logs/winograd_ansor_N%d_H%d_W%d_IC%d_OC%d_%s.log" % (
        N,
        H,
        W,
        Input_Channel,
        Output_Channel,
        time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime()),
    )
    tune_option = auto_scheduler.TuningOptions(
        num_measure_trials=Num_Tune,
        builder=auto_scheduler.LocalBuilder(
            timeout=100,
        ),
        runner=auto_scheduler.LocalRunner(
            number=5,
            repeat=3,
            timeout=50,
            min_repeat_ms=200,
            enable_cpu_cache_flush=True,
        ),
        measure_callbacks=[auto_scheduler.RecordToFile(log_dir)],
        verbose=2,
    )
    task.tune(tune_option)

    s, args = task.apply_best(log_dir)
    func = tvm.build(s, args, target, name="Winograd")
    print(tvm.lower(s, args, simple_mode=True))
    print(task.print_best(log_dir))

    return func


def gen_func_from_log(log_dir):

    task = auto_scheduler.SearchTask(
        func=winograd, args=(), target=tvm.target.Target(target),
        layout_rewrite_option=auto_scheduler.LayoutRewriteOption.REWRITE_FOR_PRE_TRANSFORMED
    )
    s, args = task.apply_best(log_dir)
    func = tvm.build(s, args, target, name="Winograd")
    print(tvm.lower(s, args, simple_mode=True))
    print(task.print_best(log_dir))

    return func


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-b", "--batch", help="the batch size, default=1", dest="b", type=int, default="1")
    parser.add_argument(
        "-H", "--height", help="the Height, default=56", dest="h", type=int, default="56")
    parser.add_argument(
        "-W", "--width", help="the Width, default=56", dest="w", type=int, default="56")
    parser.add_argument("-ic", "--input_channel",
                        help="the Input Channel, default=256", dest="ic", type=int, default="256")
    parser.add_argument("-oc", "--output_channel",
                        help="the Output Channel, default=256", dest="oc", type=int, default="256")
    parser.add_argument("-n", "--tune_num", help="the Number of Tune Steps, default=12800",
                        dest="n", type=int, default="12800")
    parser.add_argument("-l", "--log", help="get function from pre log, default=''",
                        dest="l", default="")
    parser.add_argument("-a", "--arch", help="the target architecture(x86/arm/amd), default=x86",
                        dest="a", default="x86")
    parser.add_argument("-t", "--thread", help="the Number of Thread, default=1",
                        dest="t", default="1")
    args = parser.parse_args()

    N = args.b
    Input_Channel = args.ic
    Output_Channel = args.oc
    H = args.h
    W = args.w
    Num_Tune = args.n
    H_pad = (H + 2 - (kernel_size - 1) + (block_size - 1)
             ) // block_size * block_size + (kernel_size - 1)
    W_pad = (W + 2 - (kernel_size - 1) + (block_size - 1)
             ) // block_size * block_size + (kernel_size - 1)
    nH = (H_pad - kernel_size + 1) // block_size
    nW = (W_pad - kernel_size + 1) // block_size
    Tile = N * nH * nW

    Log = args.l
    target = args.a
    if target=="x86":
        target = "llvm -mcpu=skylake-avx512"
    elif target=="arm":
        target = "llvm"
    elif target=="amd":
        target = "llvm -mcpu=core-avx2"
    else:
        target = "llvm"
    
    os.environ["OMP_NUM_THREADS"] = args.t

    print(f"Batch Size: {N}")
    print(f"Input Channel: {Input_Channel}")
    print(f"Output Channel: {Output_Channel}")
    print(f"Input Height: {H}")
    print(f"Input Width: {W}")
    print(f"Tune Num: {Num_Tune}")
    print(f"Target: {target}")
    print(f"OMP_NUM_THREADS: {os.environ.get('OMP_NUM_THREADS')}")

    if Log == "":
        func_winograd = gen_func()
    else:
        func_winograd = gen_func_from_log(Log)

    dev = tvm.device(target, 0)
    evaluator = func_winograd.time_evaluator(
        func_winograd.entry_name, dev, number=100)
    a = np.random.rand(N, H, W, Input_Channel).astype(dtype)
    b = np.random.rand(Output_Channel, Input_Channel,
                       kernel_size, kernel_size).astype(dtype)
    u = calc_kernel(b)

    U = tvm.nd.array(u, dev)
    A = tvm.nd.array(a, dev)
    O = tvm.nd.array(np.zeros((N, H, W, Output_Channel), dtype=dtype), dev)

    cost = evaluator(A, U, O).mean
    print("N=%d H=%d W=%d Input_Channel=%d Output_Channel=%d Num_Tune=%d" %
          (N, H, W, Input_Channel, Output_Channel, Num_Tune))
    print("Cost: {}".format(cost))
    print(
        "Gflops: {} {}".format(
            (
                2 * Input_Channel * Output_Channel * Tile * tile_size * tile_size
                + 2 * 2 * tile_size * tile_size * tile_size * Input_Channel * Tile
                + 2 * block_size * (tile_size + block_size) *
                tile_size * Output_Channel * Tile
            )
            * 1e-9
            / cost,
            (2 * N * Input_Channel * Output_Channel * H * W * 9) * 1e-9 / cost,
        )
    )

    print("Verifying correctness...")
    func_winograd(A, U, O)
    answer = np.zeros((N, H, W, Output_Channel), dtype=dtype)
    a = np.pad(a, ((0, 0), (1, 1), (1, 1), (0, 0)))
    for n in range(N):
        for o in range(Output_Channel):
            for i in range(Input_Channel):
                tmpa = np.zeros((H + 2, W + 2), dtype=dtype)
                for h in range(H + 2):
                    for w in range(W + 2):
                        tmpa[h][w] = a[n][h][w][i]
                tmpans = signal.correlate2d(
                    tmpa, b[o][i], mode="valid"
                )
                for h in range(H):
                    for w in range(W):
                        answer[n][h][w][o] = answer[n][h][w][o] + tmpans[h][w]
    tvm.testing.assert_allclose(O.numpy(), answer, rtol=1e-5)
    print("Done.")

