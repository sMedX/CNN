#! /usr/bin/python3

import json
import codecs
import re

basedir = "/large/datasets/retail/part16/"

with open(basedir + "test.json", "rb") as f:
    reader = codecs.getreader("utf-8")
    annotations = json.load(reader(f))

def surface(a):
    if a is None: return 0
    l, t, r, b = a
    return (r - l) * (b - t)

def intersection(a, b):
    l1, t1, r1, b1 = a
    l2, t2, r2, b2 = b

    l = max(l1, l2)
    t = max(t1, t2)

    r = min(r1, r2)
    b = min(b1, b2)

    if l > r or t > b:
        return None

    return [l, t, r, b]

def iou(a, b):
    sa = surface(a)
    sb = surface(b)
    si = surface(intersection(a, b))
    if si == 0: return 0
    return float(si) / (sa + sb - si)

assert(iou([0,0,10,10], [100,100,200,200]) == 0)
assert(iou([0,0,10,10], [5,5,15,15]) == 25. / (200. - 25.))

def match(a, b):
    return 1.0 if iou(a, b) >= 0.8 else 0.0

def optimize(labels, detected, f):
    if len(labels) == 0: return 0.
    if len(detected) == 0: return 0.

    max_value = 0
    for j in range(len(detected)):
        value = f(labels[0], detected[j]) + optimize(labels[1:], detected[:j] + detected[j+1:], f)
        if value > max_value: max_value = value

    return max_value

sum_iou = 0
sum_match = 0
sum_count = 0
tp = 0
fp = 0
tn = 0
fn = 0
for filename, labels in annotations.items():
    detected = []
    with open(basedir + filename + "_log.txt", "r") as f:
        for line in f.readlines()[1:]:
            line = line.split('[')[1]
            line = line[:-2]
            line = line.split(', ')
            line = [int(x) for x in line]
            detected.append(line)

    max_iou = optimize([l[:4] for l in labels], detected, iou)
    max_match = optimize([l[:4] for l in labels], detected, match)
    print(filename, max_iou, max_match, len(labels))

    sum_iou += max_iou
    sum_match += max_match
    sum_count += len(labels)

    tp += max_match
    fp += len(detected) - max_match
    fn += len(labels) - max_match

print(sum_iou / sum_count)
print(sum_match / sum_count)

print(tp / float(tp + fp))
print(tp / float(tp + fn))
