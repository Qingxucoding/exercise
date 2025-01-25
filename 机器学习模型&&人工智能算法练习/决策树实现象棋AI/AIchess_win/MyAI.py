import copy
import math
from ChessBoard import *


class Evaluate(object):
    # 棋子棋力得分
    single_chess_point = {
        'c': 989,   # 车
        'm': 439,   # 马
        'p': 442,   # 炮
        's': 226,   # 士
        'x': 210,   # 象
        'z': 55,    # 卒
        'j': 65536  # 将
    }
    # 红兵（卒）位置得分
    red_bin_pos_point = [
        [1, 3, 9, 10, 12, 10, 9, 3, 1],
        [18, 36, 56, 95, 118, 95, 56, 36, 18],
        [15, 28, 42, 73, 80, 73, 42, 28, 15],
        [13, 22, 30, 42, 52, 42, 30, 22, 13],
        [8, 17, 18, 21, 26, 21, 18, 17, 8],
        [3, 0, 7, 0, 8, 0, 7, 0, 3],
        [-1, 0, -3, 0, 3, 0, -3, 0, -1],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
    ]
    # 红车位置得分
    red_che_pos_point = [
        [185, 195, 190, 210, 220, 210, 190, 195, 185],
        [185, 203, 198, 230, 245, 230, 198, 203, 185],
        [180, 198, 190, 215, 225, 215, 190, 198, 180],
        [180, 200, 195, 220, 230, 220, 195, 200, 180],
        [180, 190, 180, 205, 225, 205, 180, 190, 180],
        [155, 185, 172, 215, 215, 215, 172, 185, 155],
        [110, 148, 135, 185, 190, 185, 135, 148, 110],
        [100, 115, 105, 140, 135, 140, 105, 115, 110],
        [115, 95, 100, 155, 115, 155, 100, 95, 115],
        [20, 120, 105, 140, 115, 150, 105, 120, 20]
    ]
    # 红马位置得分
    red_ma_pos_point = [
        [80, 105, 135, 120, 80, 120, 135, 105, 80],
        [80, 115, 200, 135, 105, 135, 200, 115, 80],
        [120, 125, 135, 150, 145, 150, 135, 125, 120],
        [105, 175, 145, 175, 150, 175, 145, 175, 105],
        [90, 135, 125, 145, 135, 145, 125, 135, 90],
        [80, 120, 135, 125, 120, 125, 135, 120, 80],
        [45, 90, 105, 190, 110, 90, 105, 90, 45],
        [80, 45, 105, 105, 80, 105, 105, 45, 80],
        [20, 45, 80, 80, -10, 80, 80, 45, 20],
        [20, -20, 20, 20, 20, 20, 20, -20, 20]
    ]
    # 红炮位置得分
    red_pao_pos_point = [
        [190, 180, 190, 70, 10, 70, 190, 180, 190],
        [70, 120, 100, 90, 150, 90, 100, 120, 70],
        [70, 90, 80, 90, 200, 90, 80, 90, 70],
        [60, 80, 60, 50, 210, 50, 60, 80, 60],
        [90, 50, 90, 70, 220, 70, 90, 50, 90],
        [120, 70, 100, 60, 230, 60, 100, 70, 120],
        [10, 30, 10, 30, 120, 30, 10, 30, 10],
        [30, -20, 30, 20, 200, 20, 30, -20, 30],
        [30, 10, 30, 30, -10, 30, 30, 10, 30],
        [20, 20, 20, 20, -10, 20, 20, 20, 20]
    ]
    # 红将位置得分
    red_jiang_pos_point = [
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 9750, 9800, 9750, 0, 0, 0],
        [0, 0, 0, 9900, 9900, 9900, 0, 0, 0],
        [0, 0, 0, 10000, 10000, 10000, 0, 0, 0],
    ]
    # 红相或士位置得分
    red_xiang_shi_pos_point = [
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 60, 0, 0, 0, 60, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [80, 0, 0, 80, 90, 80, 0, 0, 80],
        [0, 0, 0, 0, 0, 120, 0, 0, 0],
        [0, 0, 70, 100, 0, 100, 70, 0, 0],
    ]

    red_pos_point = {
        'z': red_bin_pos_point,
        'm': red_ma_pos_point,
        'c': red_che_pos_point,
        'j': red_jiang_pos_point,
        'p': red_pao_pos_point,
        'x': red_xiang_shi_pos_point,
        's': red_xiang_shi_pos_point
    }

    def __init__(self, team):
        self.team = team

    def get_single_chess_point(self, chess: Chess):
        if chess.team == self.team:
            return self.single_chess_point[chess.name]
        else:
            return -1 * self.single_chess_point[chess.name]

    def get_chess_pos_point(self, chess: Chess):
        red_pos_point_table = self.red_pos_point[chess.name]
        if chess.team == 'r':
            pos_point = red_pos_point_table[chess.row][chess.col]
        else:
            pos_point = red_pos_point_table[9 - chess.row][chess.col]
        if chess.team != self.team:
            pos_point *= -1
        return pos_point

    def evaluate(self, chessboard: ChessBoard):
        point = 0
        for chess in chessboard.get_chess():
            point += self.get_single_chess_point(chess)
            point += self.get_chess_pos_point(chess)
        return point


class ChessMap(object):
    def __init__(self, chessboard: ChessBoard):
        self.chess_map = copy.deepcopy(chessboard.chessboard_map)


class ChessAI(object):
    def __init__(self, computer_team):
        self.team = computer_team
        self.evaluate_class = Evaluate(self.team) #定义自己的奖励值类

    def get_min_value(self, chessboard, deepth, alpha, beta):
        if (deepth == 1):
            return self.evaluate_class.evaluate(chessboard)
            #获取所有的棋盘上的棋子对象列表
        min_num = math.inf # 定义正无穷大的浮点数，为了得到最小值，设定一个上界
        for chess in chessboard.get_chess():

            if chess.team == self.team:
                continue  # 确保是对方，若是己方则跳过
            possible_locas = chessboard.get_put_down_position(chess)
            # 获取当前被点击棋子可以落子的位置坐标

            for lo in possible_locas:
                # 遍历棋子走法,保存得分min的方案，其中lo为一个二元组
                # 保存当前棋子位置
                row_new = lo[0]
                col_new = lo[1]
                row_old = chess.row
                col_col = chess.col
                # 保存落子位置
                chess_to_set = chessboard.chessboard_map[row_new][col_new]

                # 移动棋子
                chessboard.chessboard_map[row_new][col_new] = chessboard.chessboard_map[row_old][col_col]
                # 更新要显示的图片的坐标
                chessboard.chessboard_map[row_new][col_new].update_position(row_new, col_new)
                # 清除之前的位置
                chessboard.chessboard_map[row_old][col_col] = None

                min_num = min(min_num, self.get_max_value(chessboard, deepth - 1, alpha, beta))
                # 递归

                # 回溯复原之前棋子的位置
                chessboard.chessboard_map[row_old][col_col] = chessboard.chessboard_map[row_new][col_new]
                chessboard.chessboard_map[row_old][col_col].update_position(row_old, col_col)
                chessboard.chessboard_map[row_new][col_new] = chess_to_set

                if min_num <= alpha:
                    return min_num
                beta = min(beta, min_num)
                # 修改beta值
        return min_num

    def get_max_value(self, chessboard, deepth, alpha, beta):
        if (deepth == 1):
            return self.evaluate_class.evaluate(chessboard)
        max_num = math.inf * (-1) # 定义负无穷小的浮点数，为了得到最大值，设定一个下界
        for chess in chessboard.get_chess():
            # 获取所有的棋盘上的棋子对象列表
            if chess.team != self.team:
                continue  # 确保是己方，若是对方则跳过
            possible_locas = chessboard.get_put_down_position(chess)
            for lo in possible_locas:
                # 遍历棋子走法,保存得分max的方案，其中lo为一个二元组
                # 保存当前棋子位置
                row_new = lo[0]
                col_new = lo[1]
                row_old = chess.row
                col_old = chess.col
                # 保存落子位置
                chess_to_set = chessboard.chessboard_map[row_new][col_new]

                # 移动棋子
                chessboard.chessboard_map[row_new][col_new] = chessboard.chessboard_map[row_old][col_old]
                # 更新要显示的图片的坐标
                chessboard.chessboard_map[row_new][col_new].update_position(row_new, col_new)
                # 清除之前的位置
                chessboard.chessboard_map[row_old][col_old] = None

                max_num_new = self.get_min_value(chessboard, deepth - 1, alpha, beta)
                # 递归

                # 回溯复原之前棋子的位置
                chessboard.chessboard_map[row_old][col_old] = chessboard.chessboard_map[row_new][col_new]
                chessboard.chessboard_map[row_old][col_old].update_position(row_old, col_old)
                chessboard.chessboard_map[row_new][col_new] = chess_to_set

                # 这里不可以再调用get_min_value()否则会跳不出循环
                if (max_num_new > max_num or not self.max_lo) and deepth == self.max_depth:
                    # 此时即将跳出max_depth层，需要对最佳位置进行处理
                    self.max_lo = [row_old, col_old]
                    self.max_next_lo = [row_new, col_new]
                max_num = max(max_num, max_num_new)
                if max_num > beta:
                    # 跳出循环
                    return max_num
                alpha = max(alpha, max_num)
                # 修改alpha值
        return max_num

    def get_next_step(self, chessboard: ChessBoard):
        # 存储所有棋子的所有走法中最优的成绩/棋子/落点
        self.max_next_lo = None
        self.max_lo = None
        # 设置初始值
        self.get_max_value(chessboard, self.max_depth, math.inf * (-1), math.inf)
        return [self.max_lo[0], self.max_lo[1], self.max_next_lo[0], self.max_next_lo[1]]

