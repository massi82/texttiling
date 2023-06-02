def vec_intersection(vec1, vec2, preserve_order=False):
    if preserve_order:
        set2 = frozenset(vec2)
        return [x for x in vec1 if x in set2]        
    else:
        vec1 = list(vec1)
        vec2 = list(vec2)
        vec1.sort()
        vec2.sort()
        t = []
        l1 = len(vec1)
        l2 = len(vec2)
        i = j = 0
        while i < l1 and j < l2:
            if vec1[i] == vec2[j]:
                t.append(vec1[i])
                i += 1
                j += 1
            elif vec1[i] < vec2[j]:
                i += 1
            elif vec1[i] > vec2[j]:
                j += 1
        return t
    
def vec_complement(vec1, vec2, preserve_order=False):
    if preserve_order:
        return [x for x in vec1 if x not in vec2]
    else:
        vec1 = list(vec1)
        vec2 = list(vec2)
        vec1.sort()
        vec2.sort()
        t = []
        l1 = len(vec1)
        l2 = len(vec2)
        i = j = 0
        while i < l1 and j < l2:
            if vec1[i] == vec2[j]:
                i += 1
                j += 1
            elif vec1[i] < vec2[j]:
                t.append(vec1[i])
                i += 1
            elif vec1[i] > vec2[j]:
                j += 1
        t.extend(vec1[i:])
        return list(t)
    
def find_overlap(vector1, vector2):
    min_v1, max_v1 = min(vector1), max(vector1)
    min_v2, max_v2 = min(vector2), max(vector2)

    one_in_two = [num for num in vector1 if min_v2 <= num <= max_v2]
    two_in_one = [num for num in vector2 if min_v1 <= num <= max_v1]
    overlap = one_in_two+two_in_one

    return overlap, len(one_in_two), len(two_in_one)