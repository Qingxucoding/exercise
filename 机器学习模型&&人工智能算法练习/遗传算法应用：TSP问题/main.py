import matplotlib.pyplot as picture
import matplotlib.animation as animation
import random
import math


class GeneticAlgTSP:
    def __init__(self, filename):
        # 初始化要使用的变量
        self.file = open(filename, "r")
        self.list = self.file.readlines()  # 文件每一行数据写入到list中
        self.xlist = [] # 存放city的横坐标
        self.ylist = [] # 存放city的纵坐标
        self.xylist = [] # 存放city的横纵坐标

    # 计算ab两地的距离
    def distance(self, a, b):
        s = pow(self.xylist[a - 1][1] - self.xylist[b - 1][1], 2) + pow(self.xylist[a - 1][2] - self.xylist[b - 1][2], 2)
        return math.sqrt(s)

    # 处理数据集
    def get_list(self, list):
        for file_str in list:
            file_str = file_str.strip()  # 删除字符串空白字符
            file_str = file_str.strip("\n")  # 删除字符串换行符
            tmp = file_str.split(" ")  # 按照空格进行分割
            for i in range(10):
                if '' in tmp:
                    tmp.remove('')
            temp = []
            temp.append(int(tmp[0]))
            temp.append(float(tmp[1]))
            temp.append(float(tmp[2]))
            # 保存坐标
            self.xylist.append(temp)
            self.xlist.append(temp[1])
            self.ylist.append(temp[2])

    # 评估选择函数
    def evaluate(self, path): # path为连接的city列表
        s = 0
        for i in range(1, len(path)):
            s += self.distance(path[i - 1], path[i])
        return s
        # 随机生成种群，个数为size

    def build(self, size, length):
        population = []
        for i in range(size):
            population.append(random.sample(range(1, length + 1), length))
        return population

    # 两点交叉产生下一代
    def produce(self, p1, p2):
        length = len(p1)
        sta, end = random.sample(range(1, length), 2) #随机生成起点终点
        if end < sta:
            sta, end = end, sta # 保证起点在终点左边
        child = p1[sta:end]
        c1 = [] # p1 起点左边的部位
        c2 = [] # p1 终点右边的部位
        for i in range(0, length):
            if p2[i] not in child:
                if i < sta:
                    c1.append(p2[i])
                else:
                    c2.append(p2[i])
        child = c1 + child + c2
        return child
        # 得到的实际上是p1的孩子

    # 变异函数
    def mutate(self, path):
        length = len(path)
        sta, end = random.sample(range(1, length), 2) #随机生成起点终点
        if end < sta:
            sta, end = end, sta # 保证起点在终点左边
        child = path[:sta] + path[sta:end][::-1] +path[end:]
        return child


    # 得到下一个要访问的城市
    def greed(self, vist, cur): #vist是城市的访问列表，cur是当前城市列表
        min = math.inf # 初始化正无穷数，记录存在的最小距离
        index = -1 # 表示下一个访问的城市下标
        for i in range(len(cur)):
            if cur[i] not in vist:
                for j in range(len(vist)):
                    s = self.distance(vist[j], cur[i])
                    if s < min:
                        min = s
                        index = i
        return index

    #打印初始化种群的适应度
    def printf_evaluate(self, population):
        for var in population:
            print(self.evaluate(var))
        print("building is end\n")

    # 保存动态图
    def save_animation(self, image):
        picture_ani = picture.figure(1)  # 生成动态图
        ani = animation.ArtistAnimation(picture_ani, image, interval=200, repeat_delay=1000)
        ani.save("city.gif", writer='pillow')

    def interate(self, times):
        self.get_list(self.list)
        length = len(self.xylist)
        # 生成种群，个数为20
        size = 20
        population = self.build(size, length)
        rate = 1  # 变异概率
        self.printf_evaluate(population)
        iteration = 0 #迭代计数
        image = [] #动态图采样

        while iteration < times:
            # 通过轮盘法选择亲代并杂交得到子代，选择变异后并入新种群
            new_population = []  # 新种群
            for count in range(0, 10):
                weight = []  # 转轮盘权重
                # 距离越小权重越大
                for num in range(0, len(population)):
                    weight.append(1 / self.evaluate(population[num]))
                # 选择亲代
                parent1 = random.choices(population, weight, k=1)
                parent2 = random.choices(population, weight, k=1)
                # 杂交
                child1 = self.produce(parent1[0], parent2[0])
                child2 = self.produce(parent2[0], parent1[0])
                # 变异
                if random.random() < rate:
                    child1 = self.mutate(child1)
                    child2 = self.mutate(child2)
                # 并入新种群
                new_population.append(child1)
                new_population.append(child2)

            #保存最优个体列表，并打印信息
            flag = -1
            max = math.inf # 初始化正无穷数
            # 选取最优个体并保存
            for item in range(0, len(population)):
                if self.evaluate(population[item]) < max:
                    max = self.evaluate(population[item])
                    flag = item
            temp_one = population[flag][:]
            # 打印迭代数和最优值
            print(iteration, self.evaluate(temp_one))
            population.clear()
            population = new_population[0:19]
            new_population.clear()
            population.append(temp_one)
            iteration += 1
            # 为了生成动态图需要对遍历的中间结果进行采样
            # 每一百次迭代进行一次采样
            if iteration % 100 == 0:
                xlist = []
                ylist = []
                for item_ in temp_one:
                    xlist.append(self.xylist[item_ - 1][1])
                    ylist.append(self.xylist[item_ - 1][2])
                xlist.append(self.xylist[0][1])
                ylist.append(self.xylist[0][2])
                image.append(picture.plot(xlist, ylist, marker='.', color='red', linewidth=1))
        print("the end, exit to save gif")

        flag = -1
        max = math.inf # 初始化正无穷数
        for i in range(0, len(population)):
            if self.evaluate(population[i]) < max:
                flag = i
        index = population[flag]
        print(self.evaluate(index))
        self.save_animation(image)

if __name__ == "__main__":
    filename = 'city.txt'
    city_class = GeneticAlgTSP(filename)
    sequence = city_class.interate(5000)