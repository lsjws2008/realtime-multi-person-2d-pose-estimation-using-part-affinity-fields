
def range2color(num,maxnum):
    num = num*255/maxnum
    if num < 128:
        green = 0
    else:
        green = 255- (num-128)*2
    if num < 128:
        red = 0
    else:
        red = int(num)
    if num > 128:
        blue = 0
    else:
        blue = int(255- num*2)

    if num >= 128:
        green = int(green)
    else:
        green = int(green + num*2)
    return (red, green, blue)