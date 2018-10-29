class config_set():
    def __init__(self):
        self.img_x = 800
        self.img_y = 400
        self.under_pool = 4
        self.point_num = 16
        self.paxul = 1e3
        self.link_map = [[16, 14], [14, 12], [10, 8], [8, 6], [4, 2], [2, 1], [0, 15],\
                         [15, 13], [11, 9], [9, 7], [5, 3], [3, 1]]
if __name__ == '__main__':
    co = config_set()
    print(co.link_map)