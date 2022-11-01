import json

from scipy.stats import ks_2samp
from sklearn.metrics import mean_squared_error

f = open('/home/nate/Development/SGPR/data/eigenvalues.json')
data = json.load(f)

# for every query and reference, create a ref and query list of tuples containing hash, label, and data
# loop through all query values and compare them to all ref values
# if p value is > .05 print there is a match

ref_list = []
query_list = []

for i in data:
    label = data[i]["label"]
    ref = data[i]["reference"]
    query = data[i]["query"]
    ref_list.append((i, label, ref))
    query_list.append((i, label, query))
    #result = kstest(ref, query)
    #if label == "floor" or label == "ceiling" or label == "wall" or label == "windowsill":
    #    print(label)
    #    print(result)
    #if result.pvalue > 0.05:
    #    print(label)
    #    print(result)

for i, _ in enumerate(ref_list):
    for j, _ in enumerate(query_list):
        result = ks_2samp(ref_list[i][2], query_list[j][2])
        if result.pvalue > 0.05:
            r = ref_list[i][2]
            q = query_list[j][2]
            min_size = min(len(r), len(q))
            r.reverse()
            q.reverse()
            if len(r) > min_size:
                r = r[0:min_size]
            else:
                q = q[0:min_size]
            mse = mean_squared_error(r, q)
            if mse < 40:
                print(ref_list[i][0])
                print(query_list[j][0])
                print(ref_list[i][1])
                print(query_list[j][1])
                print(result)
                print(mse)
