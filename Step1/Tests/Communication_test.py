from multiprocessing import Process

def plus():
    print("---加法进程开始执行---")
    global g_num
    g_num += 50
    print('g_num is %d' % g_num)
    print("---加法进程执行结束---")

def minus():
    print("---减法进程开始执行---")
    global g_num
    g_num -= 50
    print('g_num is %d' % g_num)
    print("---减法进程执行结束---")

g_num = 100 # 定义全局变量 g_num

if __name__ == '__main__':
    print("---主进程开始执行---")
    print('g_num is %d' % g_num)
    p1 = Process(target=plus)
    p2 = Process(target=minus)
    p1.start()
    p2.start()
    #p1.join()
    #p2.join()
    data = "%dx%d" % (300, 150)
    print(data)
    print("--主进程执行结束---")
