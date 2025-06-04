
def rmchr(text,index):
    return text[:index]+text[index+1:]

def count_rptch(text):
    maxch=(1,0)
    nowch=(0,0)
    lastch=None
    for index,i in enumerate(text):
        if lastch == i:
            nowch = (nowch[0]+1,nowch[1])
            if nowch[0]>maxch[0]:
                maxch=nowch
        else:
            nowch=(1,index)

        lastch=i

    return maxch

def remove_rptch(text,tar_len=4):
    while len(text)>tar_len:
        maxch = count_rptch(text)
        if maxch[0]<=1:
            break
        text=rmchr(text,maxch[1])
    return text