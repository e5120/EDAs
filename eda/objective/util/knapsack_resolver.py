from pulp import *

from item import Item


with open("../../../settings/knapsack.ini") as f:
    f.readline()
    f.readline()
    capacity = int(f.readline().split(",")[1])
    items = []
    for line in f:
        n, v, w = line.split(",")
        items.append(Item(n, int(v), int(w)))

value = [item.value for item in items]
weight = [item.weight for item in items]
names = [item.name for item in items]

prob = pulp.LpProblem('knapsack2', sense = pulp.LpMaximize)
# 変数の定義
xs = [pulp.LpVariable('{}'.format(x), cat='Integer', lowBound=0) for x in names]
# 目的関数
prob += pulp.lpDot(value, xs)
# 制約条件
prob += pulp.lpDot(weight, xs) <= capacity

print(prob)

status = prob.solve()
print("Status", pulp.LpStatus[status])
print([x.value() for x in xs])
print(prob.objective.value())
