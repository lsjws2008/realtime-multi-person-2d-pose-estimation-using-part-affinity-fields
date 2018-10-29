class config():
    def __init__(self):
        self.part = {0: 'right ankle', 1: 'nose', 2: 'left eye', 3: 'right eye', 4: 'left ear', 5: 'right ear',
                     6: 'left shoulder', 7: 'right shoulder', 8: 'left elbow', 9: 'right elbow',
                     10: 'left wrist', 11: 'right wrist', 12: 'left waist', 13: 'right waist',
                     14: 'left knee', 15: 'right knee', 16: 'left ankle'}

        self.lines = [[1, 2], [1, 3], [2, 4],
                      [3, 5], [6, 8], [8, 10],
                      [7, 9], [9, 11], [12, 14],
                      [14, 16], [13, 15], [15, 0]]
