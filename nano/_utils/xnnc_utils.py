a = ' 0.461|0.850|0.750|0.400 '
a = a.split('|')
a = [float(x.strip()) for x in a]
print(sum(a)/4.)