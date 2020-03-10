import math

def createMask(sigma):
    maskSize = int(math.ceil(3.0 * sigma))
    print("maskSize:", maskSize)
    mask = [None] * ((maskSize*2+1)*(maskSize*2+1))
    print("Length of mask:", len(mask))
    s = 0.0
    for a in range(-maskSize, maskSize+1):
        for b in range(-maskSize, maskSize+1):
            temp = math.exp(-(float(a*a+b*b / (2*sigma*sigma))))
            s += temp
            mask[a+maskSize+(b+maskSize)*(maskSize*2+1)] = temp

    for i in range((maskSize*2+1)*(maskSize*2+1)):
        mask[i] /= s

    return mask

print(createMask(5))

