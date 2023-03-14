from math import sin, cos, pi, log, pow
from tkinter import Tk, Canvas
from random import random, randint, choice, uniform
from multiprocessing.dummy import Pool


class HeartClass:
    def __new__(cls, *args, **kwargs):
        print('画一个小心心送给最棒最漂亮的覃，你今天也超级漂亮！！')
        print('欸嘿，今天也是想你的不知道第几天y(●ˇ∀ˇ●)y')
        return object.__new__(cls)

    def __init__(self):
        self.tk = Tk()
        self.canvas = Canvas(self.tk, background='black', height=480, width=640)
        self.heart_points = set()
        self.heart_edge_points = set()
        self.heart_inside_points = set()
        self.substitution_points = {}

    def main(self):
        self.canvas.pack()
        self.cs_dp()
        self.draw_heart()
        self.tk.mainloop()

    def cs_dp(self) -> None:
        d_xy = lambda _xy, _value: _xy - (- 0.17 * log(random()) * (_xy - _value / 2))
        for _ in range(2000):
            x, y = self.heart_equation()
            self.heart_points.add((x, y))
        for _ in range(2000):
            x, y = choice(list(self.heart_points))
            x, y = d_xy(x, 640), d_xy(y, 480)
            self.heart_inside_points.add((x, y))
        for frame in range(30):
            ratio = 10 * 2 * (2 * sin(4 * frame / 10 * pi)) / (2 * pi)
            all_points = []
            heart_halo_point = set()  # 光环的点坐标集合
            for _ in range(4000):
                x, y = self.heart_equation()  # 魔法参数
                force = - 1 / pow((pow(x - 640 / 2, 2) + pow(y - 480 / 2, 2)), 2)
                d_xy = lambda _xy, _value: _xy - (ratio * force * (_xy - _value / 2))
                x, y = d_xy(x, 640), d_xy(y, 480)
                if (x, y) not in heart_halo_point:
                    heart_halo_point.add((x, y))
                    x += randint(-14, 14)
                    y += randint(-14, 14)
                    all_points.append((x, y, choice((1, 2, 2))))

            def substitution_points(points_sets):
                for _x, _y in points_sets:
                    force = - 1 / pow((pow(_x - 640 / 2, 2) + pow(_y - 480 / 2, 2)), 0.44)
                    d_xy = lambda _xy, _value: _xy - (ratio * force * (_xy - _value / 2) + randint(-1, 1))
                    _x, _y = d_xy(_x, 640), d_xy(_y, 480)
                    all_points.append((_x, _y, randint(1, 2)))

            task = [self.heart_points, self.heart_edge_points, self.heart_inside_points]
            Pool(len(task)).map(substitution_points, task)
            self.substitution_points[frame] = all_points
        return

    def draw_heart(self, frame=0) -> None:
        self.canvas.delete('all')  # 清除上一帧绘制的点
        for x, y, size in self.substitution_points[frame % 30]:
            self.canvas.create_rectangle(x, y, x + size, y + size, width=0, fill='#FF6699')
        self.tk.after(160, self.draw_heart, frame + 1)
        return

    @staticmethod
    def heart_equation() -> int:
        ifc_factor = uniform(0, 2 * pi)
        x = int((10 * 16 * pow(sin(ifc_factor), 3)) + 640 / 2)
        y = int((0 - (13 * cos(ifc_factor) - 5 * cos(2 * ifc_factor) - 2 * cos(3 * ifc_factor) - cos(
            4 * ifc_factor)) * 10 + 480 / 2))
        return x, y


if __name__ == '__main__':
    hc = HeartClass()
    hc.main()
