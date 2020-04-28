
def find_fractions(num,eps):
    for i in range(1,100):
        for j in range(1,100):
            if abs(i/j - num) < eps:
                print(i,'/',j,str(i/j),str(abs(i/j - num)))

find_fractions(0.64385756568513,0.001)


