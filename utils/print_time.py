import time


def print_time(start, progress, epoch, total, n_epoch):
    """
    :param start: 训练开始时间
    :param progress: 当前轮的进度
    :param epoch: 总轮数
    :param total: 当前轮的总批数
    :param n_epoch: 当前第几轮
    需要打印，到目前为止已经花费的时间，训练结束需要的时间。
    """
    # print("start:%d\nprogress:%d\nepoch:%d\ntotal:%d\nn_epoch:%d\n", start, progress, epoch, total, n_epoch)
    now = time.time()
    epoch_time = now - start
    etr_time = (now - start) / (n_epoch * total + progress) * epoch * total - epoch_time

    m, s = divmod(epoch_time, 60)
    h, m = divmod(m, 60)
    print("spend time: %d:%02d:%02d" % (h, m, s))
    m, s = divmod(etr_time, 60)
    h, m = divmod(m, 60)
    print("Estimated time remaining: %d:%02d:%02d\n" % (h, m, s))


def print_test_time(start, count, total):
    now = time.time()
    epoch_time = now - start
    etr_time = (now - start) / count * (total - count)

    m, s = divmod(epoch_time, 60)
    h, m = divmod(m, 60)
    print("spend time: %d:%02d:%02d" % (h, m, s))
    m, s = divmod(etr_time, 60)
    h, m = divmod(m, 60)
    print("Estimated time remaining: %d:%02d:%02d\n" % (h, m, s))
