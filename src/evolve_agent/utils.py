def find_shortest_of_max_simple(data):
    if not data:
        return None

    # 1. First pass: Find the maximum key value.
    max_key = max(item[0] for item in data)
    
    # 2. Second pass: Create a new list of all items that have that max key.
    all_max_items = [item for item in data if item[0] == max_key]
    
    # 3. On this smaller list, find the item with the minimum length.
    return min(all_max_items, key=len)