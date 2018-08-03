import os
import subprocess

if __name__ == '__main__':
    result = []
    x = 0.0
    y = 0.0
    with open("test.pos", "r") as file:
        lines = file.readlines()
        length = len(lines)
        for line in lines:
            text = line[:-1]
            # print(text)
            p = subprocess.Popen("python deep_learning.py -predict=\"" + text +
                                 "\" -snapshot=\"../snapshot/2018-08-01_16-09-09/best_steps_100.pt\"",
                                 shell=False, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            p.wait()
            text = p.stdout.readlines()

            getLabel = False
            for t in text:
                if str(t[0:7], encoding="utf-8") == "[Label]":
                    text = str(t[8:-2], encoding="utf-8")
                    getLabel = True
                    break

            if getLabel:
                result.append(text)
                if text == "positive":
                    x += 1
                elif text == "negative":
                    y += 1
            else:
                continue

            print(text + "\tlength=" + str(length) +
                  "\tindex=" + str(x+y) +
                  "\tx=" + str(x) +
                  "\ty=" + str(y) +
                  "\tacc=" + str(x / (x + y)))


    print(x+y)
    print(x)
    print(x / (x + y))
