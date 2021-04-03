i = 0
for f1 in range(4, 7, 2):
    for f2 in range(2, f1 + 1, 2):
        for d1 in range(4, 7, 2):
            for d2 in range(2, d1 + 1, 2):

                info = str(f1) + 'c_' + str(f2) + 'c_' + str(d1) + 'd_' + str(d2) + 'd'
                print("[ " + info + " ]")
                i += 1
print("Total :", i)
