# coding:utf-8

from mpi4py import MPI
import numpy as np

# コマンドライン引数で1つめに指定された値を行列のサイズとする
# N = int(sys.argv[1])
N = 10
# マスターノードは自分自身
TaskMaster = 0

# サイズNの零行列を生成
a = np.zeros(shape=(N, N))
b = np.zeros(shape=(N, N))
c = np.zeros(shape=(N, N))


# 行番号と列番号の和をとったサイズNの行列を生成
def matgen(p):
    for i in range(0, N):
        for j in range(0, N):
            p[i][j] = i + j


comm = MPI.COMM_WORLD
# プロセス数
size = comm.Get_size()
# 今のプロセスのランク
rank = comm.Get_rank()
procName = MPI.Get_processor_name()

ta_start = MPI.Wtime()
# print ("Process %d started." % (rank))
print("Running from processor %s, rank %d out of %d processors." % (procName, rank, size))

# プロセス数が1であれば区切らない
sl = N if size == 1 else N / (size - 1)

# 区切った塊が0以下の場合は終了
assert sl >= 1

matgen(b)
# コミュニケータで同期をとる(位置について用意...)
comm.Barrier()

# マスターノードの仕事
if rank == TaskMaster:
    print("Starting to send to all workers...")
    matgen(a)
    for i in range(1, size):
        # 区切った塊の位置を指定
        offset = (i-1) * sl
        # 塊の端から端までを列とする
        row = a[offset:]
        # 送信プロセスiにタグiをつけoffsetとrowを送信
        comm.send(offset, dest=i, tag=i)
        comm.send(row, dest=i, tag=i)
        for j in range(0, sl):
            comm.send(a[j+offset, :], dest=i, tag=j+offset)
    print("All sent to Workers.")

# 同期
comm.Barrier()

# ワーカーノードの仕事
if rank != TaskMaster:
    # print ("Data Received from process %d\n" % rank)
    # マスターノードからのデータを受け取る
    offset = comm.recv(source=0, tag=rank)
    recv_data = comm.recv(source=0, tag=rank)

    for j in range(1, sl):
        c = comm.recv(source=0, tag=j + offset)
        recv_data = np.vstack((recv_data, c))

    # print ("Starting calculation from process %d...\n" % rank)

    t_start = MPI.Wtime()
    for i in range(0, sl):
        res = np.zeros(shape=(N))
        r = recv_data if sl == 1 else recv_data[i, :]
        ai = 0
        for j in range(0, N):
            q = b[:,j]
            for x in range(0, N):
                res[j] += r[x] * q[x]
            ai += 1
        send = np.vstack((send, res)) if (i > 0) else res
    t_diff = MPI.Wtime() - t_start

    print("Process %d finished in %5.4fs." % (rank, t_diff))
    # print ("Sending results to Master %d bytes.\n" % (send.nbytes))
    comm.Send([send, MPI.FLOAT], dest=0, tag=rank)

# comm.Barrier()

if rank == TaskMaster:
    res1 = np.zeros(shape=(sl, N))
    comm.Recv([res1, MPI.FLOAT], source=1, tag=1)
    k1 = np.vstack(res1)
    for i in range(2, size):
        resx = np.zeros(shape=(sl, N))
        comm.Recv([resx, MPI.FLOAT], source=i, tag=i)
        k1 = np.vstack((k1, resx))
    ta_diff = MPI.Wtime() - ta_start
    print("End.\nRes:")
    print(k1)
    print("Total Time: %5.4fs" % ta_diff)

comm.Barrier()
