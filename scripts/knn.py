import json
from sklearn.neighbors import KNeighborsClassifier

f = open('/home/nate/Development/SGPR/data/gfa_features.json')
data = json.load(f)

# TODO play with number of neighbors
neigh = KNeighborsClassifier()

reference_map = dict()
reference_scans = data["reference_scans"]
scan_idx = 0
prev_scan = ""
for scan in reference_scans:
    scan_id = scan["scan_id"]
    if prev_scan == "": prev_scan = scan_id
    if prev_scan != scan_id: scan_idx+=1
    object_map = dict()

    # populate object map
    for obj in scan:
        if obj == "scan_id": continue
        object_map[obj] = (data["reference_scans"][scan_idx][obj]["label"], data["reference_scans"][scan_idx][obj]["global_id"], data["reference_scans"][scan_idx][obj]["gfa_features"])
    reference_map[scan_id] = object_map

# parse query scans
query_map = dict()
query_scans = data["query_scans"]
scan_idx = 0
prev_scan = ""
for scan in query_scans:
    scan_id = scan["scan_id"]
    if prev_scan == "": prev_scan = scan_id
    if prev_scan != scan_id: scan_idx+=1
    reference_scan_id = scan["reference_scan_id"]
    object_map = dict()

    # populate object map
    for obj in scan:
        if obj == "scan_id" or obj == "reference_scan_id": continue
        object_map[obj] = (data["query_scans"][scan_idx][obj]["label"], data["query_scans"][scan_idx][obj]["gfa_features"], data["query_scans"][scan_idx][obj]["gfa_features"])
    query_map[scan_id] = (reference_scan_id, object_map)

X = [[0], [1], [2], [3], [7], [8], [9]]
y = [0, 0, 1, 1, 2, 2, 2]
neigh.fit(X, y)
print(neigh.predict([[1.1]]))
print(neigh.predict_proba([[0.9]]))
print(neigh.predict_proba([[6.5]]))
