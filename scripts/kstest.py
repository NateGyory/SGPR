import json

from scipy.stats import ks_2samp
from sklearn.metrics import mean_squared_error

f = open('/home/nate/Development/SGPR/data/eigenvalues_1.json')
data = json.load(f)

# Steps
# 1) first parse all reference_scans and read them into a dictionary
#   a) reference map = dict["scan_id"] = dict["ply color"] = tuple(label, eigenvalues)
#   b) query map = dict["scan_id"] = tuple(reference_scan_id, dict['ply_color']) = tuple(label,eigenvalues)

# parse reference scans
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
        object_map[obj] = (data["reference_scans"][scan_idx][obj]["label"], data["reference_scans"][scan_idx][obj]["eigenvalues"])
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
        object_map[obj] = (data["query_scans"][scan_idx][obj]["label"], data["query_scans"][scan_idx][obj]["eigenvalues"])
    query_map[scan_id] = (reference_scan_id, object_map)

# steps to test
# Test 1:
#   a) For all queries find the reference scan and run the kstest against that
#   b) Only test matching objects
def Test1():
    correct_count = 0
    for k, v in query_map.items():
        query_scan_id = k
        reference_scan_id = v[0]
        print("!!!!!!")
        print("Query scan ID: " + query_scan_id)
        print("Reference scan ID: " + reference_scan_id)
        flag = False
        for obj_k, obj_v in v[1].items():
            ply_color = obj_k
            label = obj_v[0]
            query_eigenvalues = obj_v[1]
            if ply_color in reference_map[reference_scan_id]:
                ref_eigenvalues = reference_map[reference_scan_id][ply_color][1]
                ref_label = reference_map[reference_scan_id][ply_color][0]
                result = ks_2samp(ref_eigenvalues, query_eigenvalues)
                if result.pvalue > 0.05:
                    flag = True
                    print("-----")
                    print("Query label: " + label)
                    print("Ref label: " + ref_label)
                    print("PLY color: " + ply_color)
                    print(result)
        if flag: correct_count+=1
    print(correct_count)

# Test 2:
#   a) For all queries find the reference scan and run the kstest against that
#   b) test every query object against every reference object
def Test2():
    correct_count = 0
    for k, v in query_map.items():
        query_scan_id = k
        reference_scan_id = v[0]
        print("!!!!!!")
        print("Query scan ID: " + query_scan_id)
        print("Reference scan ID: " + reference_scan_id)
        flag = False
        for obj_k, obj_v in v[1].items():
            ply_color = obj_k
            label = obj_v[0]
            query_eigenvalues = obj_v[1]
            for ref_k, ref_v in reference_map[reference_scan_id].items():
                ref_eigenvalues = ref_v[1]
                ref_label = ref_v[0]
                if len(ref_eigenvalues) > 0 and len(query_eigenvalues) > 0:
                    result = ks_2samp(ref_eigenvalues, query_eigenvalues)
                    r = ref_eigenvalues
                    q = query_eigenvalues
                    min_size = min(len(r), len(q))
                    r.reverse()
                    q.reverse()
                    if len(r) > min_size:
                        r = r[0:min_size]
                    else:
                        q = q[0:min_size]
                    mse = mean_squared_error(r, q)
                    if result.pvalue > 0.05 and mse < 40:
                        flag = True
                        print("-----")
                        print("Query label: " + label)
                        print("Ref label: " + ref_label)
                        print("PLY color: " + ply_color)
                        print("MSE: " + str(mse))
                        print(result)
                        print('\n')
        if flag: correct_count+=1
    print(correct_count)

# Test 3:
#   a) For all queries match against all references
#   b) Only test matching objects
# Steps

def Test3():
    correct_count = 0
    total_count = 0
    for k, v in query_map.items():
        total_count+=1
        query_scan_id = k
        reference_scan_id = v[0]
        print("!!!!!!")
        print("Query scan ID: " + query_scan_id)
        # list[correct_match, incorrect_match, total_match_count]
        query_ref_match_count_map = dict()
        for ref_scan_k, ref_scan_v in reference_map.items():
            #print("Analysis for reference scan ID: " + ref_scan_k)
            query_ref_match_count_map[ref_scan_k] = [0,0,0]
            #flag = False
            for obj_k, obj_v in v[1].items():
                ply_color = obj_k
                label = obj_v[0]
                query_eigenvalues = obj_v[1]
                if ply_color in reference_map[ref_scan_k]:
                    ref_eigenvalues = reference_map[ref_scan_k][ply_color][1]
                    ref_label = reference_map[ref_scan_k][ply_color][0]
                    if len(ref_eigenvalues) > 0 and len(query_eigenvalues) > 0:
                        if ref_label != label: breakpoint()
                        result = ks_2samp(ref_eigenvalues, query_eigenvalues)
                        r = ref_eigenvalues
                        q = query_eigenvalues
                        min_size = min(len(r), len(q))
                        r.reverse()
                        q.reverse()
                        if len(r) > min_size:
                            r = r[0:min_size]
                        else:
                            q = q[0:min_size]
                        mse = mean_squared_error(r, q)
                        if result.pvalue > 0.05: #and mse < 40:
                            query_ref_match_count_map[ref_scan_k][2]+=1
                            if ref_label == label:
                                query_ref_match_count_map[ref_scan_k][0]+=1
                            else:
                                query_ref_match_count_map[ref_scan_k][1]+=1
    #                        print("-----")
    #                        print("Query label: " + label)
    #                        print("Ref label: " + ref_label)
    #                        print("PLY color: " + ply_color)
    #                        print("MSE: " + str(mse))
    #                        print(result)
    #                        print('\n')
        #Print if it is the correct prediction
        max_match_tuple = ["", 0]
        for ref_scan_k, ref_scan_v in query_ref_match_count_map.items():
            if ref_scan_v[2] > max_match_tuple[1]:
                max_match_tuple[0] = ref_scan_k
                max_match_tuple[1] = ref_scan_v[2]
            print(ref_scan_k + ": correct matches: " + str(ref_scan_v[0]))
            print(ref_scan_k + ": incorrect matches: " + str(ref_scan_v[1]))
            print(ref_scan_k + ": total matches: " + str(ref_scan_v[2]))
        res = max_match_tuple[0] == reference_scan_id
        if res : correct_count+=1
        print("For: " + k + " Does " + reference_scan_id + " = " + max_match_tuple[0] + "?\n" + str(res))

    print("Correct Predictions: " + str(correct_count))
    print("Total Predictions: " + str(total_count))
    print("Accuracy: " + str(correct_count/total_count))

# Test 4:
#   a) For all queries match against all references
#   b) Test every query object against every reference object
def Test4():
    correct_count = 0
    total_count = 0
    for k, v in query_map.items():
        total_count+=1
        query_scan_id = k
        reference_scan_id = v[0]
        print("!!!!!!")
        print("Query scan ID: " + query_scan_id)
        query_ref_match_count_map = dict()
        for ref_scan_k, ref_scan_v in reference_map.items():
            number_of_objs = len(ref_scan_v)
            query_ref_match_count_map[ref_scan_k] = [0,0,0,number_of_objs,[],[]]
            #flag = False
            for obj_k, obj_v in v[1].items():
                ply_color = obj_k
                label = obj_v[0]
                query_eigenvalues = obj_v[1]
                for ref_k, ref_v in reference_map[ref_scan_k].items():
                    ref_eigenvalues = ref_v[1]
                    ref_label = ref_v[0]
                    if len(ref_eigenvalues) > 0 and len(query_eigenvalues) > 0:
                        result = ks_2samp(ref_eigenvalues, query_eigenvalues)
                        r = ref_eigenvalues
                        q = query_eigenvalues
                        min_size = min(len(r), len(q))
                        r.reverse()
                        q.reverse()
                        if len(r) > min_size:
                            r = r[0:min_size]
                        else:
                            q = q[0:min_size]
                        mse = mean_squared_error(r, q)
                        if result.pvalue > 0.05 and mse < 40:
                            query_ref_match_count_map[ref_scan_k][2]+=1
                            if ref_label == label:
                                if ref_scan_k != reference_scan_id:
                                    query_ref_match_count_map[ref_scan_k][4].append(label)
                                query_ref_match_count_map[ref_scan_k][0]+=1
                            else:
                                query_ref_match_count_map[ref_scan_k][1]+=1
                                query_ref_match_count_map[ref_scan_k][5].append((label, ref_label))
    #                        print("-----")
    #                        print("Query label: " + label)
    #                        print("Ref label: " + ref_label)
    #                        print("PLY color: " + ply_color)
    #                        print("MSE: " + str(mse))
    #                        print(result)
    #                        print('\n')
        #Print if it is the correct prediction
        max_match_tuple = ["", 0]
        for ref_scan_k, ref_scan_v in query_ref_match_count_map.items():
            if ref_scan_v[2] > max_match_tuple[1]:
                max_match_tuple[0] = ref_scan_k
                max_match_tuple[1] = ref_scan_v[2]
            print("---------------------------------------")
            print(ref_scan_k + ": total objects in scene: " + str(ref_scan_v[3]))
            print(ref_scan_k + ": correct matches: " + str(ref_scan_v[0]))
            print(ref_scan_k + ": incorrect matches: " + str(ref_scan_v[1]))
            print(ref_scan_k + ": total matches: " + str(ref_scan_v[2]))
            print("Labels which were correctly matched but from a wrong scene: ")
            print('\n'.join(map(str, ref_scan_v[4])))
            print("Labels which were incorrectly matched: ")
            print('\n'.join(map(str, ref_scan_v[5])))
            print("---------------------------------------")
        res = max_match_tuple[0] == reference_scan_id
        if res: correct_count+=1
        print("For: " + k + " Does " + reference_scan_id + " = " + max_match_tuple[0] + "?\n" + str(res))

    print("Correct Predictions: " + str(correct_count))
    print("Total Predictions: " + str(total_count))
    print("Accuracy: " + str(correct_count/total_count))

# Test 5:
#   a) For all queries match against all references
#   b) Test every semantic label if they match
def Test5():
    correct_count = 0
    total_count = 0
    for k, v in query_map.items():
        total_count+=1
        query_scan_id = k
        reference_scan_id = v[0]
        print("!!!!!!")
        print("Query scan ID: " + query_scan_id)
        print("Number of objects in the scene: " + str(len(v[1])))
        query_ref_match_count_map = dict()
        for ref_scan_k, ref_scan_v in reference_map.items():
            number_of_objs = len(ref_scan_v)
            query_ref_match_count_map[ref_scan_k] = [0,0,0,number_of_objs,[],[]]
            #flag = False
            for obj_k, obj_v in v[1].items():
                ply_color = obj_k
                label = obj_v[0]
                query_eigenvalues = obj_v[1]
                for ref_k, ref_v in reference_map[ref_scan_k].items():
                    ref_eigenvalues = ref_v[1]
                    ref_label = ref_v[0]
                    if label == ref_label:
                        if len(ref_eigenvalues) > 0 and len(query_eigenvalues) > 0:
                            result = ks_2samp(ref_eigenvalues, query_eigenvalues)
                            r = ref_eigenvalues
                            q = query_eigenvalues
                            min_size = min(len(r), len(q))
                            r.reverse()
                            q.reverse()
                            if len(r) > min_size:
                                r = r[0:min_size]
                            else:
                                q = q[0:min_size]
                            mse = mean_squared_error(r, q)
                            if result.pvalue > 0.05:# and mse < 40:
                                query_ref_match_count_map[ref_scan_k][2]+=1
                                if ref_label == label:
                                    if ref_scan_k != reference_scan_id:
                                        query_ref_match_count_map[ref_scan_k][4].append(label)

                                    query_ref_match_count_map[ref_scan_k][0]+=1
                                else:
                                    query_ref_match_count_map[ref_scan_k][1]+=1
                                    query_ref_match_count_map[ref_scan_k][5].append((label, ref_label))

    #                            print("-----")
    #                            print("Query label: " + label)
    #                            print("Ref label: " + ref_label)
    #                            print("PLY color: " + ply_color)
    #                            print("MSE: " + str(mse))
    #                            print(result)
    #                            print('\n')
        #Print if it is the correct prediction
        max_match_tuple = ["", 0]
        for ref_scan_k, ref_scan_v in query_ref_match_count_map.items():
            if ref_scan_v[2] > max_match_tuple[1]:
                max_match_tuple[0] = ref_scan_k
                max_match_tuple[1] = ref_scan_v[2]
            print("---------------------------------------")
            print(ref_scan_k + ": total objects in scene: " + str(ref_scan_v[3]))
            print(ref_scan_k + ": correct matches: " + str(ref_scan_v[0]))
            print(ref_scan_k + ": incorrect matches: " + str(ref_scan_v[1]))
            print(ref_scan_k + ": total matches: " + str(ref_scan_v[2]))
            print("Labels which were correctly matched but from a wrong scene: ")
            print('\n'.join(map(str, ref_scan_v[4])))
            print("---------------------------------------")
        res = max_match_tuple[0] == reference_scan_id
        if res: correct_count+=1
        print("For: " + k + " Does " + reference_scan_id + " = " + max_match_tuple[0] + "?\n" + str(res))

    print("Correct Predictions: " + str(correct_count))
    print("Total Predictions: " + str(total_count))
    print("Accuracy: " + str(correct_count/total_count))
Test5()
